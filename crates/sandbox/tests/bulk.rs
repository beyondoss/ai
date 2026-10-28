//! Bulk copy over a transport written in the test (`sh` on this host): the wire shape of an upload,
//! the staged path with a small chunk, and the compensating action when a command fails.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "sdk_conformance/local_sh.rs"]
mod local_sh;

use std::os::unix::fs::PermissionsExt as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use beyond_sandbox::bulk::{self, BulkError, Sources};
use beyond_sandbox::{Exec, ExecResult};

/// LocalSh that records every command and can fail the nth one that carries stdin.
struct Spy {
    inner: local_sh::LocalSh,
    calls: Mutex<Vec<(String, Vec<String>, usize)>>,
    fail_stdin_call: Option<usize>,
    stdin_calls: Mutex<usize>,
}

impl Spy {
    fn new(fail_stdin_call: Option<usize>) -> Arc<Self> {
        Arc::new(Self {
            inner: local_sh::LocalSh,
            calls: Mutex::new(Vec::new()),
            fail_stdin_call,
            stdin_calls: Mutex::new(0),
        })
    }
    fn calls(&self) -> Vec<(String, Vec<String>, usize)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl Exec for Spy {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.calls
            .lock()
            .unwrap()
            .push((program.into(), args.to_vec(), 0));
        self.inner.run(program, args, cwd, timeout).await
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        self.calls
            .lock()
            .unwrap()
            .push((program.into(), args.to_vec(), stdin.len()));
        let n = {
            let mut n = self.stdin_calls.lock().unwrap();
            *n += 1;
            *n
        };
        if self.fail_stdin_call == Some(n) {
            return Err(std::io::Error::other("exec endpoint unreachable"));
        }
        self.inner
            .run_with_stdin(program, args, cwd, timeout, stdin)
            .await
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: beyond_sandbox::exec::ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.calls
            .lock()
            .unwrap()
            .push((program.into(), args.to_vec(), 0));
        self.inner
            .run_streaming(program, args, cwd, timeout, on_chunk)
            .await
    }

    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &beyond_sandbox::exec::JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        self.inner
            .run_streaming_tracked(program, args, cwd, hooks)
            .await
    }

    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: beyond_sandbox::exec::ChunkSink<'_>,
        on_stderr: beyond_sandbox::exec::ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.calls
            .lock()
            .unwrap()
            .push((program.into(), args.to_vec(), 0));
        self.inner
            .run_attached(program, args, cwd, on_stdout, on_stderr)
            .await
    }
}

fn tree(root: &std::path::Path) {
    std::fs::create_dir_all(root.join("sub/empty")).unwrap();
    std::fs::write(root.join("a.txt"), b"alpha\n").unwrap();
    std::fs::write(
        root.join("sub/b.bin"),
        (0..=255u8).cycle().take(20_000).collect::<Vec<_>>(),
    )
    .unwrap();
    std::fs::set_permissions(
        root.join("sub/b.bin"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    std::os::unix::fs::symlink("a.txt", root.join("link")).unwrap();
}

#[tokio::test]
async fn a_small_tree_is_one_request_and_comes_back_whole() {
    let dir = tempfile::tempdir().unwrap();
    let (src, remote, back) = (
        dir.path().join("src"),
        dir.path().join("remote"),
        dir.path().join("back"),
    );
    tree(&src);
    let spy = Spy::new(None);
    let t = bulk::upload(&*spy, Sources::Dir(src.clone()), remote.to_str().unwrap())
        .await
        .unwrap();
    assert_eq!((t.files, t.dirs, t.links, t.bytes), (2, 2, 1, 20_006));
    let calls = spy.calls();
    assert_eq!(calls.len(), 1, "one command, the archive on stdin");
    assert_eq!(calls[0].0, "sh");
    assert_eq!(calls[0].1[0], "-c");
    assert_eq!(calls[0].1[2], "sh");
    assert_eq!(calls[0].1[3], remote.to_str().unwrap());
    assert_eq!(calls[0].1[4], "");
    assert!(calls[0].2 > 20_000);

    let d = bulk::download(&*spy, remote.to_str().unwrap(), &back)
        .await
        .unwrap();
    assert_eq!((d.files, d.dirs, d.links, d.bytes), (2, 2, 1, 20_006));
    assert_eq!(std::fs::read(back.join("a.txt")).unwrap(), b"alpha\n");
    assert_eq!(
        std::fs::read_link(back.join("link")).unwrap().to_str(),
        Some("a.txt")
    );
    assert!(back.join("sub/empty").is_dir());
    let mode = std::fs::metadata(back.join("sub/b.bin"))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600);
}

#[tokio::test]
async fn a_larger_archive_is_staged_in_chunks_and_extracted_once() {
    let dir = tempfile::tempdir().unwrap();
    let (src, remote) = (dir.path().join("src"), dir.path().join("remote"));
    tree(&src);
    let spy = Spy::new(None);
    bulk::upload_chunked(&*spy, Sources::Dir(src), remote.to_str().unwrap(), 4096)
        .await
        .unwrap();
    let calls = spy.calls();
    let appends = calls.iter().filter(|c| c.2 > 0).count();
    // 20 KB of data, headers and the end block in 4 KiB chunks.
    assert!((6..=9).contains(&appends), "{appends} appends");
    // The last command extracts the stage and has no stdin.
    let last = calls.last().unwrap();
    assert_eq!(last.2, 0);
    assert!(
        last.1[4].starts_with(&format!("{}.upload.", remote.display())),
        "{:?}",
        last.1
    );
    assert_eq!(std::fs::read(remote.join("a.txt")).unwrap(), b"alpha\n");
    let left: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(left.iter().all(|n| !n.contains("upload")), "{left:?}");
}

#[tokio::test]
async fn a_failed_append_removes_the_stage_and_reports_the_transport() {
    let dir = tempfile::tempdir().unwrap();
    let (src, remote) = (dir.path().join("src"), dir.path().join("remote"));
    tree(&src);
    let spy = Spy::new(Some(3));
    let err = bulk::upload_chunked(&*spy, Sources::Dir(src), remote.to_str().unwrap(), 4096)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unreachable"), "{err}");
    let calls = spy.calls();
    let last = calls.last().unwrap();
    assert_eq!(
        (last.0.as_str(), last.1[0].as_str(), last.1[1].as_str()),
        ("rm", "-f", "--")
    );
    assert!(!remote.exists(), "nothing was extracted");
    let left: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
    assert_eq!(left.len(), 1, "only src is left");
}

#[tokio::test]
async fn a_destination_that_is_a_file_is_reported_and_left_alone() {
    let dir = tempfile::tempdir().unwrap();
    let (src, file) = (dir.path().join("src"), dir.path().join("f"));
    tree(&src);
    std::fs::write(&file, b"mine").unwrap();
    let err = bulk::upload(
        &local_sh::LocalSh,
        Sources::Dir(src),
        file.to_str().unwrap(),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(&err, BulkError::Command(m) if m.contains("Not a directory")),
        "{err}"
    );
    assert_eq!(std::fs::read(&file).unwrap(), b"mine");
}

#[tokio::test]
async fn bad_requests_send_nothing() {
    let spy = Spy::new(None);
    for dest in ["relative", "/", "//"] {
        let r = bulk::upload(&*spy, Sources::Paths(vec![]), dest).await;
        assert!(
            matches!(r, Err(BulkError::InvalidRequest(_))),
            "{dest}: {r:?}"
        );
    }
    let r = bulk::download(&*spy, "relative", std::path::Path::new("/tmp")).await;
    assert!(matches!(r, Err(BulkError::InvalidRequest(_))));
    assert!(spy.calls().is_empty());
}

#[tokio::test]
async fn a_list_of_sources_lands_by_base_name() {
    let dir = tempfile::tempdir().unwrap();
    let (src, remote) = (dir.path().join("src"), dir.path().join("remote"));
    tree(&src);
    bulk::upload(
        &local_sh::LocalSh,
        Sources::Paths(vec![src.join("a.txt"), src.join("sub")]),
        remote.to_str().unwrap(),
    )
    .await
    .unwrap();
    assert!(remote.join("a.txt").is_file());
    assert!(remote.join("sub/b.bin").is_file());
    assert!(!remote.join("link").exists());
}
