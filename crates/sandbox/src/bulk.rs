//! Bulk copy: `upload` and `download` of whole trees as a tar stream over raw exec
//! (FILES-OVER-EXEC.md section 5.8).
//!
//! Layer 2 like [`crate::Files`]: built only on [`CommandRunner`]. The archive is written and read
//! by the `tar` crate, so the client's machine needs no `tar` binary; the sandbox needs `sh` and a
//! `tar`, GNU or busybox. The local side of a download is untrusted input and is extracted by
//! rules that never write through a link (`extract`).

mod extract;

use std::fs::{self, File};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use crate::exec::{CommandRunner, ExecOutput};

use extract::{Extractor, extract_stream};

/// The smallest chunk: one tar block.
const BLOCK: usize = 512;

/// Raw tar bytes per upload request. 16 MiB is 21.3 MB of base64: under the 64 MiB an `execd`
/// accepts, and equal to the one-shot response cap. A call holds about two chunks.
pub const BULK_CHUNK_BYTES: usize = 16 * 1024 * 1024;
/// The deadline of the extraction command and of the download stream: a hung `tar` ends.
pub const BULK_TIMEOUT: Duration = Duration::from_millis(600_000);
/// The deadline of a stage append and of a cleanup, the helpers' usual.
const STAGE_TIMEOUT: Duration = Duration::from_millis(120_000);
/// The stderr a download keeps for its error detail.
pub const BULK_STDERR_BYTES: usize = 64 * 1024;

/// UPLOAD_SCRIPT, appendix G of FILES-OVER-EXEC.md.
const UPLOAD_SCRIPT: &str = r#"d="$1"
s="$2"
if [ -e "$d" ] && [ ! -d "$d" ]; then
  echo "Not a directory (os error 20)" >&2
  [ -z "$s" ] || rm -f -- "$s"
  exit 1
fi
new=
if [ ! -d "$d" ]; then
  mkdir -p -- "$d" || { [ -z "$s" ] || rm -f -- "$s"; exit 1; }
  new=1
fi
umask 0
if cd -- "$d"; then
  if [ -n "$s" ]; then tar -xf "$s"; else tar -xf -; fi && { [ -z "$s" ] || rm -f -- "$s"; exit 0; }
fi
[ -z "$s" ] || rm -f -- "$s"
cd /
[ -z "$new" ] || rm -rf -- "$d"
exit 1"#;

/// DOWNLOAD_SCRIPT, appendix H.
const DOWNLOAD_SCRIPT: &str = r#"if [ ! -e "$1" ]; then
  echo "No such file or directory (os error 2)" >&2
  exit 1
fi
if [ -d "$1" ]; then
  exec tar -cf - -C "$1" .
fi
exec tar -cf - -C "$(dirname -- "$1")" "./$(basename -- "$1")""#;

/// STAGE_SCRIPT, appendix I.
const STAGE_SCRIPT: &str = r#"f="$1"
if [ "$2" -eq 0 ]; then
  mkdir -p -- "$(dirname -- "$f")" || exit 1
  set -C
  : > "$f" || exit 1
  set +C
fi
n=$(wc -c < "$f") || exit 1
if [ "$n" -ne "$2" ]; then
  echo "stage is $n bytes, expected $2" >&2
  exit 1
fi
cat >> "$f" || exit 1
n=$(wc -c < "$f")
if [ "$n" -ne $(($2 + $3)) ]; then
  echo "short write: $n of $(($2 + $3)) bytes in the stage" >&2
  exit 1
fi"#;

/// What a copy moved: members that are regular files, directories (not the root), links, and the
/// bytes of file content.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Transfer {
    pub files: u64,
    pub dirs: u64,
    pub links: u64,
    pub bytes: u64,
}

/// Why a copy failed, in the terms of FILES-OVER-EXEC.md section 3.1.
#[derive(Debug)]
pub enum BulkError {
    /// The arguments cannot form a request; nothing was sent (`invalid_request`).
    InvalidRequest(String),
    /// An archive that is malformed or unsafe (`protocol`).
    Archive(String),
    /// A command ran and failed; the message is `upload|download PATH: LINE` and is classified by
    /// its text (section 3.3).
    Command(String),
    /// The transport or the local filesystem failed; classified by its kind and text.
    Io(std::io::Error),
}

impl std::fmt::Display for BulkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(m) | Self::Archive(m) | Self::Command(m) => f.write_str(m),
            Self::Io(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for BulkError {}

impl From<std::io::Error> for BulkError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// What to upload.
#[derive(Debug, Clone)]
pub enum Sources {
    /// A directory: its contents land directly in the destination.
    Dir(PathBuf),
    /// Files and directories: each lands in the destination under its base name.
    Paths(Vec<PathBuf>),
}

/// The detail of a failed command: the first line of trimmed stderr, extended to the first line that
/// names an OS error (section 3.3).
fn first_line(stderr: &str) -> String {
    const NAMES: [&str; 6] = [
        "no such file or directory",
        "directory nonexistent",
        "nonexistent directory",
        "is a directory",
        "not a directory",
        "permission denied",
    ];
    let t = stderr.trim();
    if t.is_empty() {
        return "command failed".into();
    }
    let mut end = None;
    let mut off = 0;
    for line in t.split_inclusive('\n') {
        let e = off + line.trim_end_matches('\n').len();
        if NAMES.iter().any(|n| line.to_ascii_lowercase().contains(n)) {
            return t[..e].to_string();
        }
        end.get_or_insert(e);
        off += line.len();
    }
    t[..end.unwrap_or(t.len())].to_string()
}

fn sh(script: &str, args: &[&str]) -> Vec<String> {
    let mut v = vec!["-c".to_string(), script.to_string(), "sh".to_string()];
    v.extend(args.iter().map(|a| (*a).to_string()));
    v
}

fn remote_path(what: &str, p: &str, not_root: bool) -> Result<(), BulkError> {
    if !p.starts_with('/') {
        return Err(BulkError::InvalidRequest(format!(
            "{what} must be absolute: {p:?}"
        )));
    }
    if not_root && p.trim_end_matches('/').is_empty() {
        return Err(BulkError::InvalidRequest(format!("{what} must not be /")));
    }
    if p.contains('\0') {
        return Err(BulkError::InvalidRequest(format!("{what} has a NUL")));
    }
    Ok(())
}

fn local_err(path: &Path, e: std::io::Error) -> BulkError {
    BulkError::Io(std::io::Error::new(
        e.kind(),
        format!("{}: {e}", path.display()),
    ))
}

// ---- upload -----------------------------------------------------------------------------------

struct Chunk {
    bytes: Vec<u8>,
    last: bool,
}

/// The archive's sink: collects what the `tar` writer produces and hands out full chunks (blocking:
/// it runs on a blocking thread).
struct ChunkWriter {
    buf: Vec<u8>,
    chunk: usize,
    tx: tokio::sync::mpsc::Sender<Chunk>,
}

impl ChunkWriter {
    /// Send a full chunk once there is one. `Err` is the receiver gone: the upload failed or was
    /// dropped, and the walk stops.
    fn flush(&mut self, force_last: bool) -> std::io::Result<()> {
        while self.buf.len() >= self.chunk || force_last {
            let n = self.buf.len().min(self.chunk);
            let last = force_last && self.buf.len() <= self.chunk;
            let rest = self.buf.split_off(n);
            let bytes = std::mem::replace(&mut self.buf, rest);
            self.tx
                .blocking_send(Chunk { bytes, last })
                .map_err(|_| std::io::Error::other("upload aborted"))?;
            if last {
                return Ok(());
            }
        }
        Ok(())
    }
}

impl std::io::Write for ChunkWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(bytes);
        self.flush(false)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A file's content, exactly as many bytes as its header says: a file that shrinks while it is read
/// fails the upload (a short member would corrupt the archive), one that grows is cut.
struct Exact {
    file: File,
    left: u64,
    path: PathBuf,
}

impl std::io::Read for Exact {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.left == 0 {
            return Ok(0);
        }
        let want = buf.len().min(self.left as usize);
        let n = self.file.read(&mut buf[..want])?;
        if n == 0 {
            return Err(std::io::Error::other(format!(
                "{}: changed while being archived",
                self.path.display()
            )));
        }
        self.left -= n as u64;
        Ok(n)
    }
}

struct Builder {
    tar: tar::Builder<ChunkWriter>,
    counts: Transfer,
}

/// The upload's own error back out of the `tar` writer's `io::Error`.
fn upload_err(path: &Path, e: std::io::Error) -> BulkError {
    if e.to_string() == "upload aborted" || e.to_string().contains("changed while being archived") {
        return BulkError::Io(e);
    }
    local_err(path, e)
}

impl Builder {
    fn add(&mut self, name: &[u8], path: &Path) -> Result<(), BulkError> {
        let meta = fs::symlink_metadata(path).map_err(|e| local_err(path, e))?;
        let ft = meta.file_type();
        if std::str::from_utf8(name).is_err() {
            return Err(BulkError::InvalidRequest(format!(
                "{}: the name is not valid UTF-8",
                path.display()
            )));
        }
        let mut h = tar::Header::new_gnu();
        h.set_mode(meta.permissions().mode() & 0o777);
        h.set_mtime(u64::try_from(meta.mtime()).unwrap_or(0));
        h.set_uid(0);
        h.set_gid(0);
        let os = |b: &[u8]| std::ffi::OsStr::from_bytes(b).to_owned();
        if ft.is_dir() {
            h.set_entry_type(tar::EntryType::Directory);
            h.set_size(0);
            let mut n = name.to_vec();
            n.push(b'/');
            self.tar
                .append_data(&mut h, Path::new(&os(&n)), std::io::empty())
                .map_err(|e| upload_err(path, e))?;
            self.counts.dirs += 1;
            let mut names: Vec<_> = fs::read_dir(path)
                .map_err(|e| local_err(path, e))?
                .map(|r| r.map(|d| d.file_name()))
                .collect::<Result<_, _>>()
                .map_err(|e| local_err(path, e))?;
            names.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            for n in names {
                let mut child = name.to_vec();
                if !child.is_empty() {
                    child.push(b'/');
                }
                child.extend_from_slice(n.as_bytes());
                self.add(&child, &path.join(&n))?;
            }
        } else if ft.is_symlink() {
            h.set_entry_type(tar::EntryType::Symlink);
            h.set_mode(0o777);
            h.set_size(0);
            let target = fs::read_link(path).map_err(|e| local_err(path, e))?;
            self.tar
                .append_link(&mut h, Path::new(&os(name)), target)
                .map_err(|e| upload_err(path, e))?;
            self.counts.links += 1;
        } else if ft.is_file() {
            h.set_entry_type(tar::EntryType::Regular);
            h.set_size(meta.len());
            let file = File::open(path).map_err(|e| local_err(path, e))?;
            let body = Exact {
                file,
                left: meta.len(),
                path: path.to_path_buf(),
            };
            self.tar
                .append_data(&mut h, Path::new(&os(name)), body)
                .map_err(|e| upload_err(path, e))?;
            self.counts.files += 1;
            self.counts.bytes += meta.len();
        } else {
            let what = if ft.is_fifo() {
                "a FIFO"
            } else if ft.is_socket() {
                "a socket"
            } else {
                "a device"
            };
            return Err(BulkError::InvalidRequest(format!(
                "{}: {what} cannot be uploaded",
                path.display()
            )));
        }
        Ok(())
    }
}

fn build(
    sources: Sources,
    chunk: usize,
    tx: tokio::sync::mpsc::Sender<Chunk>,
) -> Result<Transfer, BulkError> {
    let mut b = Builder {
        tar: tar::Builder::new(ChunkWriter {
            buf: Vec::new(),
            chunk,
            tx,
        }),
        counts: Transfer::default(),
    };
    match sources {
        Sources::Dir(dir) => {
            let meta = fs::metadata(&dir).map_err(|e| local_err(&dir, e))?;
            if !meta.is_dir() {
                return Err(BulkError::Io(std::io::Error::other(format!(
                    "{}: Not a directory (os error 20)",
                    dir.display()
                ))));
            }
            let mut names: Vec<_> = fs::read_dir(&dir)
                .map_err(|e| local_err(&dir, e))?
                .map(|r| r.map(|d| d.file_name()))
                .collect::<Result<_, _>>()
                .map_err(|e| local_err(&dir, e))?;
            names.sort_by(|a, c| a.as_bytes().cmp(c.as_bytes()));
            for n in names {
                b.add(n.as_bytes(), &dir.join(&n))?;
            }
        }
        Sources::Paths(paths) => {
            for p in &paths {
                let Some(name) = p.file_name() else {
                    return Err(BulkError::InvalidRequest(format!(
                        "{}: no base name",
                        p.display()
                    )));
                };
                b.add(name.as_bytes(), p)?;
            }
        }
    }
    // The end-of-archive blocks, then the last chunk.
    let mut w = b.tar.into_inner().map_err(BulkError::Io)?;
    w.flush(true).map_err(BulkError::Io)?;
    Ok(b.counts)
}

fn failed(what: &str, dest: &str, out: &ExecOutput) -> BulkError {
    BulkError::Command(format!("{what} {dest}: {}", first_line(&out.stderr_text())))
}

/// Upload `sources` into the sandbox directory `remote_dir` (created if missing), with chunks of
/// [`BULK_CHUNK_BYTES`].
pub async fn upload(
    exec: &dyn CommandRunner,
    sources: Sources,
    remote_dir: &str,
) -> Result<Transfer, BulkError> {
    upload_chunked(exec, sources, remote_dir, BULK_CHUNK_BYTES).await
}

/// [`upload`] with a chosen chunk size (at least one block), so a small tree can walk the staged path.
pub async fn upload_chunked(
    exec: &dyn CommandRunner,
    sources: Sources,
    remote_dir: &str,
    chunk_bytes: usize,
) -> Result<Transfer, BulkError> {
    remote_path("remote_dir", remote_dir, true)?;
    let dest = remote_dir.trim_end_matches('/').to_string();
    let chunk = chunk_bytes.max(BLOCK);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Chunk>(1);
    let producer = tokio::task::spawn_blocking(move || build(sources, chunk, tx));

    let result = send(exec, &dest, &mut rx).await;
    drop(rx);
    let built = producer
        .await
        .map_err(|e| BulkError::Io(std::io::Error::other(e.to_string())))?;
    match (result, built) {
        // The local side failed first (an unreadable file, a name): its error is the cause.
        (_, Err(e)) if !matches!(&e, BulkError::Io(io) if io.to_string() == "upload aborted") => {
            Err(e)
        }
        (Err(e), _) => Err(e),
        (Ok(()), Ok(counts)) => Ok(counts),
        (Ok(()), Err(e)) => Err(e),
    }
}

/// The remote half: one command for an archive of one chunk, a stage and an extraction otherwise.
async fn send(
    exec: &dyn CommandRunner,
    dest: &str,
    rx: &mut tokio::sync::mpsc::Receiver<Chunk>,
) -> Result<(), BulkError> {
    let Some(first) = rx.recv().await else {
        // The producer ended without an archive: its error is reported by the caller.
        return Ok(());
    };
    if first.last {
        let out: ExecOutput = exec
            .run_with_stdin(
                "sh",
                &sh(UPLOAD_SCRIPT, &[dest, ""]),
                None,
                BULK_TIMEOUT,
                &first.bytes,
            )
            .await?
            .into();
        return if out.code == 0 {
            Ok(())
        } else {
            Err(failed("upload", dest, &out))
        };
    }

    let stage = format!("{dest}.upload.{}", suffix());
    let r = staged(exec, dest, &stage, first, rx).await;
    if r.is_err() {
        // The compensating action: the scripts remove the stage on the failures they see.
        let _ = exec
            .run(
                "rm",
                &["-f".into(), "--".into(), stage],
                None,
                STAGE_TIMEOUT,
            )
            .await;
    }
    r
}

async fn staged(
    exec: &dyn CommandRunner,
    dest: &str,
    stage: &str,
    first: Chunk,
    rx: &mut tokio::sync::mpsc::Receiver<Chunk>,
) -> Result<(), BulkError> {
    let mut offset = 0usize;
    let mut next = Some(first);
    while let Some(c) = next.take() {
        if !c.bytes.is_empty() {
            let (o, l) = (offset.to_string(), c.bytes.len().to_string());
            let out: ExecOutput = exec
                .run_with_stdin(
                    "sh",
                    &sh(STAGE_SCRIPT, &[stage, &o, &l]),
                    None,
                    STAGE_TIMEOUT,
                    &c.bytes,
                )
                .await?
                .into();
            if out.code != 0 {
                return Err(failed("upload", dest, &out));
            }
            offset += c.bytes.len();
        }
        if c.last {
            let out: ExecOutput = exec
                .run("sh", &sh(UPLOAD_SCRIPT, &[dest, stage]), None, BULK_TIMEOUT)
                .await?
                .into();
            return if out.code == 0 {
                Ok(())
            } else {
                Err(failed("upload", dest, &out))
            };
        }
        next = rx.recv().await;
    }
    // The producer stopped before the archive was complete.
    Err(BulkError::Io(std::io::Error::other("upload aborted")))
}

fn suffix() -> String {
    let mut b = [0u8; 8];
    // Unpredictable, so a preplanted name cannot redirect the stage.
    if getrandom::fill(&mut b).is_err() {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        b = t.to_le_bytes();
    }
    hex::encode(b)
}

// ---- download ---------------------------------------------------------------------------------

/// The receiving end of the download's pipe as a `Read`, for the extraction thread.
struct ChanReader {
    rx: std::sync::mpsc::Receiver<Vec<u8>>,
    cur: Vec<u8>,
    pos: usize,
}

impl std::io::Read for ChanReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while self.pos >= self.cur.len() {
            match self.rx.recv() {
                Ok(v) => {
                    self.cur = v;
                    self.pos = 0;
                }
                Err(_) => return Ok(0),
            }
        }
        let n = buf.len().min(self.cur.len() - self.pos);
        buf[..n].copy_from_slice(&self.cur[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Chunks of stdout waiting for the extraction thread: the bound on what a slow disk holds back.
const PIPE_CHUNKS: usize = 8;

/// Push one stdout chunk at the extract thread. A full pipe blocks until that thread takes it.
/// On a multi-thread runtime the wait is [`tokio::task::block_in_place`], so the worker can run
/// other tasks; on a current-thread runtime `block_in_place` panics, and the extract thread is
/// already off the runtime, so the worker waits directly.
fn send_chunk(tx: &std::sync::mpsc::SyncSender<Vec<u8>>, bytes: Vec<u8>) {
    let multithreaded = tokio::runtime::Handle::try_current()
        .is_ok_and(|handle| handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread);
    if multithreaded {
        tokio::task::block_in_place(|| {
            let _ = tx.send(bytes);
        });
    } else {
        let _ = tx.send(bytes);
    }
}

/// Download `remote_path` (a directory's contents, or one file) into `local_dir`, created if missing.
pub async fn download(
    exec: &dyn CommandRunner,
    remote_path: &str,
    local_dir: &Path,
) -> Result<Transfer, BulkError> {
    remote_path_check(remote_path)?;
    fs::create_dir_all(local_dir).map_err(|e| local_err(local_dir, e))?;
    // The command's stdout goes down a bounded channel to a blocking thread that reads it as a tar
    // archive and extracts it as it arrives.
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(PIPE_CHUNKS);
    let tx = Mutex::new(Some(tx));
    let bad = std::sync::Arc::new(tokio::sync::Notify::new());
    let worker = {
        let (bad, root) = (bad.clone(), local_dir.to_path_buf());
        tokio::task::spawn_blocking(move || {
            let mut ex = Extractor::new(&root);
            let r = extract_stream(
                ChanReader {
                    rx,
                    cur: Vec::new(),
                    pos: 0,
                },
                &mut ex,
            );
            if r.is_err() {
                bad.notify_one();
            }
            r.and_then(|()| ex.finish())
        })
    };
    let feed = |b: &[u8]| {
        let guard = tx.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(t) = &*guard {
            // A thread that has stopped has dropped its end: the notify below ends the call.
            // The send blocks when the extract thread is behind. On a multi-thread runtime that
            // wait must not pin the worker, or every other task stalls for a slow disk.
            send_chunk(t, b.to_vec());
        }
    };
    let args = sh(DOWNLOAD_SCRIPT, &[remote_path]);
    let stderr = Mutex::new(Vec::<u8>::new());

    let out: Result<ExecOutput, BulkError> = {
        let on_err = |b: &[u8]| {
            let mut s = stderr
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let room = BULK_STDERR_BYTES.saturating_sub(s.len());
            s.extend_from_slice(&b[..b.len().min(room)]);
        };
        let run = exec.run_attached("sh", &args, None, &feed, &on_err);
        tokio::select! {
            r = tokio::time::timeout(BULK_TIMEOUT, run) => match r {
                Ok(r) => r.map(ExecOutput::from).map_err(BulkError::Io),
                Err(_) => Err(BulkError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("download {remote_path}: timed out"),
                ))),
            },
            // A bad member closes the stream: dropping `run` kills the command.
            () = bad.notified() => Ok(ExecOutput::default()),
        }
    };
    // No more input: the thread sees the end of the archive (or its absence).
    drop(
        tx.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take(),
    );
    let extracted = worker
        .await
        .map_err(|e| BulkError::Io(std::io::Error::other(e.to_string())))?;
    let out = out?;
    match extracted {
        // An unsafe or local failure is the cause whatever the command said.
        Err(e) if !matches!(&e, BulkError::Archive(m) if !m.starts_with("unsafe archive member")) => {
            Err(e)
        }
        // A command that failed leaves an archive that is cut short: the command's message is the report.
        other if out.code != 0 => {
            drop(other);
            let err =
                String::from_utf8_lossy(&stderr.into_inner().unwrap_or_default()).into_owned();
            Err(BulkError::Command(format!(
                "download {remote_path}: {}",
                first_line(&err)
            )))
        }
        other => other,
    }
}

fn remote_path_check(p: &str) -> Result<(), BulkError> {
    remote_path("remote_path", p, false)
}

/// Extract an archive already in memory under `local_dir`: the local half of a download, for a
/// transport that delivered the bytes some other way and for tests.
pub fn extract(archive: &[u8], local_dir: &Path) -> Result<Transfer, BulkError> {
    fs::create_dir_all(local_dir).map_err(|e| local_err(local_dir, e))?;
    let mut ex = Extractor::new(local_dir);
    extract_stream(archive, &mut ex)?;
    ex.finish()
}
