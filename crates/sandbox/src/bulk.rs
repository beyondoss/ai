//! Bulk copy: `upload` and `download` of whole trees as a tar stream over raw exec
//! (FILES-OVER-EXEC.md section 5.8).
//!
//! Layer 2 like [`crate::Files`]: built only on [`CommandRunner`]. The archive is written and read
//! by this module (`archive`), so the client's machine needs no `tar`; the sandbox needs `sh` and a
//! `tar`, GNU or busybox. The local side of a download is untrusted input and is extracted by
//! rules that never write through a link (`extract`).

mod archive;
mod extract;

use std::fs::{self, File};
use std::io::Read as _;
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use crate::exec::{CommandRunner, ExecOutput};

use archive::{BLOCK, Entry, Parser, pad_to_block, write_end, write_header};
use extract::Extractor;

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

/// Writes the archive and hands out chunks (blocking: it runs on a blocking thread).
struct Builder {
    buf: Vec<u8>,
    chunk: usize,
    tx: tokio::sync::mpsc::Sender<Chunk>,
    counts: Transfer,
}

impl Builder {
    /// Send a full chunk once there is one. `Err` is the receiver gone: the upload failed or was
    /// dropped, and the walk stops.
    fn flush(&mut self, force_last: bool) -> Result<(), BulkError> {
        while self.buf.len() >= self.chunk || force_last {
            let n = self.buf.len().min(self.chunk);
            let last = force_last && self.buf.len() <= self.chunk;
            let rest = self.buf.split_off(n);
            let bytes = std::mem::replace(&mut self.buf, rest);
            self.tx
                .blocking_send(Chunk { bytes, last })
                .map_err(|_| BulkError::Io(std::io::Error::other("upload aborted")))?;
            if last {
                return Ok(());
            }
        }
        Ok(())
    }

    fn add(&mut self, name: &[u8], path: &Path) -> Result<(), BulkError> {
        let meta = fs::symlink_metadata(path).map_err(|e| local_err(path, e))?;
        let ft = meta.file_type();
        let mut e = Entry {
            name: name.to_vec(),
            kind: b'0',
            mode: meta.permissions().mode() & 0o777,
            size: 0,
            mtime: u64::try_from(meta.mtime()).unwrap_or(0),
            link: Vec::new(),
        };
        if std::str::from_utf8(name).is_err() {
            return Err(BulkError::InvalidRequest(format!(
                "{}: the name is not valid UTF-8",
                path.display()
            )));
        }
        if ft.is_dir() {
            e.kind = b'5';
            write_header(&mut self.buf, &e);
            self.counts.dirs += 1;
            self.flush(false)?;
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
            e.kind = b'2';
            e.mode = 0o777;
            e.link = fs::read_link(path)
                .map_err(|e| local_err(path, e))?
                .as_os_str()
                .as_bytes()
                .to_vec();
            write_header(&mut self.buf, &e);
            self.counts.links += 1;
            self.flush(false)?;
        } else if ft.is_file() {
            e.size = meta.len();
            write_header(&mut self.buf, &e);
            let mut f = File::open(path).map_err(|e| local_err(path, e))?;
            let mut left = e.size;
            while left > 0 {
                let at = self.buf.len();
                let want = left.min(64 * 1024) as usize;
                self.buf.resize(at + want, 0);
                let got = f
                    .read(&mut self.buf[at..])
                    .map_err(|e| local_err(path, e))?;
                self.buf.truncate(at + got);
                if got == 0 {
                    return Err(BulkError::Io(std::io::Error::other(format!(
                        "{}: changed while being archived",
                        path.display()
                    ))));
                }
                left -= got as u64;
                self.counts.bytes += got as u64;
                self.flush(false)?;
            }
            pad_to_block(&mut self.buf, e.size);
            self.counts.files += 1;
            self.flush(false)?;
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
        buf: Vec::new(),
        chunk,
        tx,
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
    write_end(&mut b.buf);
    b.flush(true)?;
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

/// The extractor and the parser behind one lock, because a streaming sink is `Fn`.
struct Pipe {
    parser: Parser,
    extractor: Extractor,
    error: Option<BulkError>,
}

impl Pipe {
    fn feed(&mut self, bytes: &[u8]) {
        if self.error.is_none()
            && let Err(e) = self.parser.feed(bytes, &mut self.extractor)
        {
            self.error = Some(e);
        }
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
    let pipe = Mutex::new(Pipe {
        parser: Parser::new(),
        extractor: Extractor::new(local_dir),
        error: None,
    });
    let args = sh(DOWNLOAD_SCRIPT, &[remote_path]);
    let stderr = Mutex::new(Vec::<u8>::new());
    let bad = tokio::sync::Notify::new();

    let out: ExecOutput = if exec.attaches().await {
        let on_out = |b: &[u8]| {
            let mut p = pipe
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            p.feed(b);
            if p.error.is_some() {
                bad.notify_one();
            }
        };
        let on_err = |b: &[u8]| {
            let mut s = stderr
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let room = BULK_STDERR_BYTES.saturating_sub(s.len());
            s.extend_from_slice(&b[..b.len().min(room)]);
        };
        let run = exec.run_attached("sh", &args, None, &on_out, &on_err);
        tokio::select! {
            r = tokio::time::timeout(BULK_TIMEOUT, run) => match r {
                Ok(r) => r?.into(),
                Err(_) => return Err(BulkError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("download {remote_path}: timed out"),
                ))),
            },
            // An unsafe member closes the stream: dropping `run` kills the command.
            () = bad.notified() => ExecOutput::default(),
        }
    } else {
        let r: ExecOutput = exec.run("sh", &args, None, BULK_TIMEOUT).await?.into();
        pipe.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .feed(&r.stdout);
        *stderr
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = r.stderr.clone();
        ExecOutput {
            stdout: Vec::new(),
            ..r
        }
    };

    let mut p = pipe
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(e) = p.error {
        return Err(e);
    }
    if out.code != 0 {
        let err = String::from_utf8_lossy(&stderr.into_inner().unwrap_or_default()).into_owned();
        return Err(BulkError::Command(format!(
            "download {remote_path}: {}",
            first_line(&err)
        )));
    }
    p.parser.finish()?;
    p.extractor.finish()
}

fn remote_path_check(p: &str) -> Result<(), BulkError> {
    remote_path("remote_path", p, false)
}

/// Extract an archive already in memory under `local_dir`: the local half of a download, for a
/// transport that delivered the bytes some other way and for tests.
pub fn extract(archive: &[u8], local_dir: &Path) -> Result<Transfer, BulkError> {
    fs::create_dir_all(local_dir).map_err(|e| local_err(local_dir, e))?;
    let mut parser = Parser::new();
    let mut ex = Extractor::new(local_dir);
    parser.feed(archive, &mut ex)?;
    parser.finish()?;
    ex.finish()
}
