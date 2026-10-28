//! Remote **writes** at real sizes: the content travels on stdin, in one call.
//!
//! A remote write once shipped the whole file as one base64 argv entry, and Linux caps a single
//! argv string at 128 KiB (`MAX_ARG_STRLEN`) — so every remote `write`/`edit` of a file over ~96 KiB
//! failed with `Argument list too long`. The mock here runs its commands on this (Linux) host, so it
//! enforces that same limit: these tests fail on the old code for the real reason, not a simulated one.
//!
//! The assertion that matters most is [`a_write_whose_stdin_never_arrives_fails_instead_of_emptying_the_file`]:
//! a transport that silently drops stdin must never turn a write into an empty file.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::sync::Arc;

use beyond_agent::exec_endpoint::{
    DEFAULT_MAX_RESPONSE_BYTES, ExecCell, ExecTarget, HttpExecRunner,
};
use beyond_agent::tools::exec::{CommandRunner, RealRunner};
use beyond_agent::tools::fs::FsBackend;
use beyond_agent::tools::fs::shell::{Capabilities, ShellFs};
use beyond_agent::tools::{ToolConfig, default_registry_with_config};
use common::exec_mock::ExecMock;
use serde_json::{Value, json};

use crate::common;

/// The host runner with every `stdin` swallowed: the command runs, and sees an empty input.
struct SwallowsStdin;

#[async_trait::async_trait]
impl CommandRunner for SwallowsStdin {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: std::time::Duration,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        RealRunner.run(program, args, cwd, timeout).await
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: std::time::Duration,
        _stdin: &[u8],
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        RealRunner.run(program, args, cwd, timeout).await
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: std::time::Duration,
        on_chunk: beyond_agent::tools::exec::ChunkSink<'_>,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        RealRunner
            .run_streaming(program, args, cwd, timeout, on_chunk)
            .await
    }

    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &beyond_agent::tools::exec::JobHooks<'_>,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        RealRunner
            .run_streaming_tracked(program, args, cwd, hooks)
            .await
    }

    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: beyond_agent::tools::exec::ChunkSink<'_>,
        on_stderr: beyond_agent::tools::exec::ChunkSink<'_>,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        RealRunner
            .run_attached(program, args, cwd, on_stdout, on_stderr)
            .await
    }
}

/// 200 KiB — comfortably past the 128 KiB single-argv cap even before base64's 4/3 growth.
const BIG: usize = 200 * 1024;
/// `ShellFs`'s raw bytes per chunk on the fallback path (base64s to 64 KiB).
const CHUNK: usize = 48 * 1024;

/// Every byte value, in an order that isn't periodic at any chunk boundary.
fn pattern(len: usize) -> Vec<u8> {
    let mut x: u32 = 0x9E37_79B9;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 24) as u8
        })
        .collect()
}

/// No `<name>.tmp.*` may survive a write, successful or not.
fn assert_no_temp_left(dir: &Path) {
    let leftovers: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp."))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temp files left behind: {leftovers:?}"
    );
}

/// A request's one-line shape, so an assertion failure does not print 270 KiB of base64.
fn summary(reqs: &[Value]) -> Vec<String> {
    reqs.iter()
        .map(|r| {
            let longest = r["args"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::len)
                .max()
                .unwrap_or(0);
            format!(
                "{} (longest arg {longest} B, stdin: {})",
                r["command"],
                r["stdin"].as_str().map_or(0, str::len)
            )
        })
        .collect()
}

#[tokio::test]
async fn a_200k_write_is_one_call_and_byte_exact() {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start(dir.path()).await;
    let fs = ShellFs::connect(Arc::new(HttpExecRunner::new(&mock.url).unwrap()))
        .await
        .unwrap();
    let path = dir.path().join("big.bin");
    let bytes = pattern(BIG);
    let before = mock.bodies().len();
    fs.write_bytes(&path, bytes.clone()).await.unwrap();

    let writes = mock.bodies().split_off(before);
    assert_eq!(writes.len(), 1, "one round trip: {:?}", summary(&writes));
    assert!(
        writes[0]["stdin"].is_string(),
        "the content must travel on stdin: {:?}",
        summary(&writes)
    );
    assert!(std::fs::read(&path).unwrap() == bytes, "not byte-exact");
    assert_no_temp_left(dir.path());
}

#[tokio::test]
async fn a_write_whose_stdin_never_arrives_fails_instead_of_emptying_the_file() {
    // A transport that closes the command's stdin: the far command sees empty input. The write must
    // fail — not commit zero bytes.
    let dir = tempfile::tempdir().unwrap();
    let fs = ShellFs::connect(Arc::new(SwallowsStdin)).await.unwrap();
    let path = dir.path().join("precious.txt");
    std::fs::write(&path, "old content\n").unwrap();

    let err = fs.write_bytes(&path, pattern(BIG)).await.unwrap_err();
    assert!(err.to_string().contains("short write"), "{err}");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "old content\n",
        "a failed write must leave the old file exactly as it was"
    );
    assert_no_temp_left(dir.path());
}

#[tokio::test]
async fn a_write_is_byte_exact_at_every_chunk_boundary() {
    // Over `RealRunner`, so the only variable is the write. The empty file is the edge where a write
    // most easily sends nothing at all and never creates the file.
    let dir = tempfile::tempdir().unwrap();
    let fs = ShellFs::with_capabilities(Arc::new(RealRunner), Capabilities::default());
    for len in [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK, BIG] {
        let path = dir.path().join(format!("f-{len}"));
        let bytes = pattern(len);
        fs.write_bytes(&path, bytes.clone())
            .await
            .unwrap_or_else(|e| panic!("len={len}: {e}"));
        assert!(
            std::fs::read(&path).unwrap() == bytes,
            "len={len}: not byte-exact"
        );
    }
    assert_no_temp_left(dir.path());
}

#[tokio::test]
async fn the_write_and_edit_tools_handle_a_200k_file_on_a_v1_endpoint() {
    // The tools the model actually calls, end to end: the defect surfaced as `write`/`edit` failing.
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start(dir.path()).await;
    let runner: Arc<dyn CommandRunner> = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
    let reg = default_registry_with_config(&ToolConfig {
        fs_backend: Some(
            Arc::new(ShellFs::connect(runner.clone()).await.unwrap()) as Arc<dyn FsBackend>
        ),
        command_runner: Some(runner),
        ..ToolConfig::new()
    });
    let path = dir.path().join("big.txt");
    let line = "the quick brown fox jumps over the lazy dog 0123456789\n";
    let content = format!("{}MARKER\n", line.repeat(BIG / line.len()));

    let run = |tool: &'static str, input: Value| {
        let reg = &reg;
        async move {
            reg.get(tool)
                .unwrap()
                .run(input)
                .await
                .unwrap_or_else(|e| panic!("{tool}: {e}"))
        }
    };
    run("write", json!({ "path": path, "content": content })).await;
    assert_eq!(std::fs::read_to_string(&path).unwrap(), content);

    run(
        "edit",
        json!({ "path": path, "old_string": "MARKER", "new_string": "EDITED" }),
    )
    .await;
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        content.replace("MARKER", "EDITED")
    );
    assert_no_temp_left(dir.path());
}

#[tokio::test]
async fn a_cell_delegates_stdin_to_whatever_it_currently_holds() {
    let dir = tempfile::tempdir().unwrap();
    let cell = ExecCell::new();
    let echo = |runner: Arc<dyn CommandRunner>| async move {
        runner
            .run_with_stdin(
                "cat",
                &[],
                None,
                std::time::Duration::from_secs(10),
                b"through the cell",
            )
            .await
            .unwrap()
            .stdout_text()
            .into_owned()
    };
    // Empty: this host.
    assert_eq!(echo(cell.runner()).await, "through the cell");
    // Pointed at a target: that target.
    let mock = ExecMock::start(dir.path()).await;
    cell.set(Some(
        ExecTarget::http(&mock.url, &[], DEFAULT_MAX_RESPONSE_BYTES)
            .await
            .unwrap(),
    ));
    assert_eq!(echo(cell.runner()).await, "through the cell");
}
