//! A job's output spool: where its standard output and error go, and how execd reads them.
//!
//! A job's stdout and stderr are two files, `out` and `err`, opened by execd with `O_APPEND` and
//! inherited by the job's processes ([`execd_jobs::output_sink`], S1). There is no pipe, so there
//! is no reader whose death turns the job's next write into `SIGPIPE`: execd can be stopped,
//! restarted or replaced and the job writes on. execd reads each file from its own offset
//! ([`Tail`]) into the job's ring, which is where the offsets, the stderr labels and the retention
//! still live (`ring.rs`); a restarted execd opens the same files again and reads what they hold.
//!
//! **Wakeups.** A file gives no readiness, so one `inotify` instance ([`Hub`]) watches every job's
//! files for `IN_MODIFY` and wakes the job's tail: a write reaches the ring as fast as it reached
//! it through a pipe, with no polling. A tail always reads to the end before it waits, and a wake
//! that finds nothing to read is harmless, so a lost or doubled event costs nothing; a queue
//! overflow wakes every tail.
//!
//! **Bounds.** Bytes execd has read are returned to the filesystem (`FALLOC_FL_PUNCH_HOLE`, in
//! 64 KiB steps), so a spool holds what execd has not yet read plus under one step per stream. The
//! file's length is the stream's byte count; its blocks are the backlog. A spool that cannot be
//! punched (a filesystem without it) keeps everything until the job is removed; that is logged
//! once.
//!
//! **Where.** With `EXECD_STATE_DIR`, `<state>/spool/<mark>/`: it outlives the process, and
//! recovery finds it by the job's mark. Without, under the jobs directory, which dies with execd
//! (nothing could adopt the job anyway). The directory is `0700`, the files `0600`, created
//! `O_EXCL` and opened `O_NOFOLLOW`, and it is deleted when the job is removed.

use std::collections::HashMap;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{self, Write as _};
use std::os::fd::{AsFd, AsRawFd, RawFd};
use std::os::unix::fs::{DirBuilderExt as _, FileExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use nix::fcntl::{FallocateFlags, OFlag, fallocate};
use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify, WatchDescriptor};
use nix::unistd::{Whence, lseek};
use tokio::io::unix::AsyncFd;
use tokio::sync::Notify;

use crate::proc::locked;

/// Bytes read at once, and the step in which read bytes are returned to the filesystem.
const STEP: usize = 64 * 1024;

/// One job's spool directory.
#[derive(Debug)]
pub(crate) struct Spool {
    dir: PathBuf,
}

/// The two files a job's output goes to, opened for the job: a write end per stream.
pub(crate) struct Writers {
    pub out: File,
    pub err: File,
}

fn open_flags() -> i32 {
    OFlag::O_NOFOLLOW.bits()
}

impl Spool {
    /// Create `<base>/<mark>` and its two files, `0700` and `0600`, and open them for writing in
    /// append mode.
    pub fn create(base: &Path, mark: &str) -> io::Result<(Self, Writers)> {
        DirBuilder::new().recursive(true).mode(0o700).create(base)?;
        let dir = base.join(mark);
        DirBuilder::new().mode(0o700).create(&dir)?;
        let spool = Self { dir };
        let open = |name: &str| {
            OpenOptions::new()
                .append(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(open_flags())
                .open(spool.dir.join(name))
        };
        let w = (|| {
            Ok(Writers {
                out: open("out")?,
                err: open("err")?,
            })
        })();
        if w.is_err() {
            spool.remove();
        }
        w.map(|w| (spool, w))
    }

    /// The spool at `dir`, which a previous execd made (the ledger names it), if it is still there
    /// and is a real directory.
    pub fn at(dir: &Path) -> Option<Self> {
        let meta = fs::symlink_metadata(dir).ok()?;
        meta.is_dir().then(|| Self {
            dir: dir.to_path_buf(),
        })
    }

    /// The directory, which the ledger records.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Where this job's ring keeps its chunk files: with the spool, so a restart finds them.
    pub fn ring_dir(&self) -> PathBuf {
        self.dir.join("ring")
    }

    /// Delete the directory and the files in it. Idempotent.
    pub fn remove(&self) {
        let _ = fs::remove_dir_all(&self.dir);
    }

    /// Delete every spool under `base` that is not one of `keep`: the ones no job in the ledger
    /// names (a job that ended or was removed before the restart).
    pub fn retain(base: &Path, keep: &[&Path]) {
        let Ok(entries) = fs::read_dir(base) else {
            return;
        };
        for e in entries.flatten() {
            let Ok(meta) = fs::symlink_metadata(e.path()) else {
                continue;
            };
            if meta.is_dir() && !keep.contains(&e.path().as_path()) {
                let _ = fs::remove_dir_all(e.path());
            }
        }
    }
}

/// Where a tail starts reading.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Start {
    /// A new job.
    Zero,
    /// A job a previous execd left without a journal.
    Data,
    /// The cursors of `out` and `err` ([`execd_jobs::resume`]).
    At([u64; 2]),
}

/// One stream's read side.
struct Stream {
    file: File,
    /// The next byte to read.
    off: u64,
    /// Everything before this is returned to the filesystem.
    punched: u64,
}

impl Stream {
    fn open(path: &Path, start: Start, which: usize) -> io::Result<Self> {
        // Read and write: the write is for `fallocate`, which punches what has been read.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(open_flags())
            .open(path)?;
        let off = match start {
            Start::Zero => 0,
            // The journal is gone: what is not a hole is what the last execd had not yet read (or
            // read in the step it had not yet returned). Bytes may repeat; none is lost.
            Start::Data => match lseek(file.as_fd(), 0, Whence::SeekData) {
                Ok(o) => u64::try_from(o).unwrap_or(0),
                // Nothing but a hole: everything was read.
                Err(_) => file.metadata()?.len(),
            },
            // The kernel's `resume`: exactly after the last byte consumed.
            Start::At(c) => c[which],
        };
        Ok(Self {
            file,
            off,
            punched: off,
        })
    }

    /// Return what is read, up to the last whole step, to the filesystem.
    fn punch(&mut self, warned: &mut bool) {
        let upto = self.off - self.off % STEP as u64;
        if upto < self.punched + STEP as u64 {
            return;
        }
        let r = fallocate(
            self.file.as_fd(),
            FallocateFlags::FALLOC_FL_PUNCH_HOLE | FallocateFlags::FALLOC_FL_KEEP_SIZE,
            self.punched as i64,
            (upto - self.punched) as i64,
        );
        match r {
            Ok(()) => self.punched = upto,
            Err(e) => {
                if !*warned {
                    *warned = true;
                    crate::log_warn!("cannot return read job output to the filesystem: {e}");
                }
                // Not retried at every read: the next attempt starts a step on.
                self.punched = upto;
            }
        }
    }
}

/// The read side of a job's spool: both files, each from its own offset.
pub(crate) struct Tail {
    streams: [Stream; 2],
    buf: Vec<u8>,
    warned: bool,
    /// Woken when a file of this job is written.
    pub wake: Arc<Notify>,
    hub: Arc<Hub>,
    wds: [WatchDescriptor; 2],
}

impl Tail {
    /// Open the job's files for reading and start watching them.
    pub fn open(hub: &Arc<Hub>, spool: &Spool, start: Start) -> io::Result<Self> {
        let streams = [
            Stream::open(&spool.dir.join("out"), start, 0)?,
            Stream::open(&spool.dir.join("err"), start, 1)?,
        ];
        let wake = Arc::new(Notify::new());
        let wds = [
            hub.watch(&spool.dir.join("out"), &wake)?,
            hub.watch(&spool.dir.join("err"), &wake)?,
        ];
        Ok(Self {
            streams,
            buf: vec![0u8; STEP],
            warned: false,
            wake,
            hub: hub.clone(),
            wds,
        })
    }

    /// The next byte of `out` and of `err` to read.
    pub fn offsets(&self) -> [u64; 2] {
        [self.streams[0].off, self.streams[1].off]
    }

    /// Read both files to their ends, a buffer from each in turn so neither starves the other, and
    /// hand each piece to `sink` (`stderr` for the `err` file). `sink` returns `false` to stop (the
    /// job is gone); then so does this. Yields to the runtime after every round, so a job
    /// writing flat out cannot hold the one thread.
    pub async fn drain(&mut self, mut sink: impl FnMut(&[u8], bool) -> bool) -> bool {
        loop {
            let mut any = false;
            for (i, s) in self.streams.iter_mut().enumerate() {
                let n = match s.file.read_at(&mut self.buf, s.off) {
                    Ok(n) => n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => 0,
                };
                if n == 0 {
                    continue;
                }
                any = true;
                s.off += n as u64;
                if !sink(&self.buf[..n], i == 1) {
                    return false;
                }
                s.punch(&mut self.warned);
            }
            if !any {
                return true;
            }
            tokio::task::yield_now().await;
        }
    }
}

impl Drop for Tail {
    fn drop(&mut self) {
        for wd in self.wds {
            self.hub.unwatch(wd);
        }
    }
}

const JOURNAL: &str = "journal";
/// Bytes of the journal's header: the two cursors at its start.
const HEADER: usize = 16;

/// The unfinished chunk of a job's ring, kept on disk beside it: what execd consumed from the two
/// files and has not yet written to a chunk, in the order it ring-ed it, with the cursors of the two
/// files at its start. A restarted execd rebuilds the ring from the chunk files and this
/// ([`execd_jobs::resume`], [`execd_jobs::replay_skip`]), so a client's offset names the same byte
/// before and after. Appended before the bytes read are returned to the filesystem, so a byte is
/// always in a spool file or in the journal. Replaced whole (written aside and renamed) whenever
/// the ring flushes a chunk, so it never holds more than the unfinished chunk plus one piece.
pub(crate) struct Journal {
    dir: PathBuf,
    file: File,
    /// The cursors of `out` and `err` at the first byte held.
    start: [u64; 2],
    /// What the file holds after the header, as it was ring-ed.
    pieces: std::collections::VecDeque<(bool, Vec<u8>)>,
    held: usize,
    warned: bool,
    buf: Vec<u8>,
}

impl Journal {
    /// A journal in `dir` holding the last `ring_tail` bytes of `pieces`, which follow `start`
    /// (a new job's: nothing, from zero).
    pub fn create(
        dir: &Path,
        start: [u64; 2],
        pieces: Vec<(bool, Vec<u8>)>,
        ring_tail: usize,
    ) -> io::Result<Self> {
        let held = pieces.iter().map(|p| p.1.len()).sum();
        let mut j = Self {
            dir: dir.to_path_buf(),
            // Replaced by `rewrite`.
            file: File::open("/dev/null")?,
            start,
            pieces: pieces.into(),
            held,
            warned: false,
            buf: Vec::new(),
        };
        j.trim_to(ring_tail);
        j.rewrite()?;
        Ok(j)
    }

    /// Drop the front until `ring_tail` bytes are held: those went to a chunk, so they are behind
    /// the cursors now.
    fn trim_to(&mut self, ring_tail: usize) {
        let mut drop_n = self.held.saturating_sub(ring_tail);
        self.held -= drop_n;
        while drop_n > 0 {
            let Some((e, p)) = self.pieces.front_mut() else {
                break;
            };
            let n = drop_n.min(p.len());
            self.start[usize::from(*e)] += n as u64;
            p.drain(..n);
            drop_n -= n;
            if p.is_empty() {
                self.pieces.pop_front();
            }
        }
    }

    /// `data` was ring-ed (from stderr when `stderr`) and the ring's unfinished chunk is now
    /// `ring_tail` bytes: append it, or, when the ring flushed a chunk, replace the journal with
    /// the unfinished chunk's last `ring_tail` bytes.
    pub fn record(&mut self, stderr: bool, data: &[u8], ring_tail: usize) {
        self.pieces.push_back((stderr, data.to_vec()));
        self.held += data.len();
        let r = if ring_tail == self.held {
            self.buf.clear();
            self.buf.push(u8::from(stderr));
            self.buf
                .extend_from_slice(&(data.len() as u32).to_le_bytes());
            self.buf.extend_from_slice(data);
            self.file.write_all(&self.buf)
        } else {
            self.trim_to(ring_tail);
            self.rewrite()
        };
        if let Err(e) = r
            && !self.warned
        {
            self.warned = true;
            crate::log_warn!("cannot journal a job's output; a crash may lose its offsets: {e}");
        }
    }

    /// Replace the file with the header and the pieces, atomically.
    fn rewrite(&mut self) -> io::Result<()> {
        let tmp = self.dir.join("journal.new");
        let _ = fs::remove_file(&tmp);
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(open_flags())
            .open(&tmp)?;
        let mut out = Vec::with_capacity(HEADER + self.held + 5 * self.pieces.len());
        out.extend_from_slice(&self.start[0].to_le_bytes());
        out.extend_from_slice(&self.start[1].to_le_bytes());
        for (e, p) in &self.pieces {
            out.push(u8::from(*e));
            out.extend_from_slice(&(p.len() as u32).to_le_bytes());
            out.extend_from_slice(p);
        }
        f.write_all(&out)?;
        drop(f);
        fs::rename(&tmp, self.dir.join(JOURNAL))?;
        self.file = OpenOptions::new()
            .append(true)
            .custom_flags(open_flags())
            .open(self.dir.join(JOURNAL))?;
        Ok(())
    }

    /// What a previous execd journalled in `dir`: the cursors at its start and its pieces. A piece
    /// the crash cut short is not there (its bytes are still in the spool files). `None`: no
    /// journal, or not one of ours.
    pub fn load(dir: &Path) -> Option<Saved> {
        let bytes = fs::read(dir.join(JOURNAL)).ok()?;
        let head = bytes.get(..HEADER)?;
        let start = [
            u64::from_le_bytes(head[..8].try_into().ok()?),
            u64::from_le_bytes(head[8..].try_into().ok()?),
        ];
        let mut at = HEADER;
        let mut pieces = Vec::new();
        while let Some(h) = bytes.get(at..at + 5) {
            let len = u32::from_le_bytes(h[1..].try_into().ok()?) as usize;
            let Some(data) = bytes.get(at + 5..at + 5 + len) else {
                break;
            };
            pieces.push((h[0] == 1, data.to_vec()));
            at += 5 + len;
        }
        Some((start, pieces))
    }
}

/// A journal as read back: the cursors at its start and its pieces (`true`: stderr).
pub(crate) type Saved = ([u64; 2], Vec<(bool, Vec<u8>)>);

/// `inotify`'s descriptor, for the runtime.
struct Fd(Inotify);

impl AsRawFd for Fd {
    fn as_raw_fd(&self) -> RawFd {
        self.0.as_fd().as_raw_fd()
    }
}

struct Inner {
    fd: AsyncFd<Fd>,
    wakes: Mutex<HashMap<WatchDescriptor, Arc<Notify>>>,
}

impl Inner {
    /// Wake the tail of every file written since the last call.
    fn dispatch(&self) {
        loop {
            match self.fd.get_ref().0.read_events() {
                Ok(events) => {
                    let wakes = locked(&self.wakes);
                    for e in events {
                        if e.mask.contains(AddWatchFlags::IN_Q_OVERFLOW) {
                            // Events were lost: every tail reads to its end, which is all a wake
                            // asks of it.
                            wakes.values().for_each(|n| n.notify_one());
                        } else if let Some(n) = wakes.get(&e.wd) {
                            n.notify_one();
                        }
                    }
                }
                Err(_) => return, // EAGAIN: drained
            }
        }
    }
}

/// The one `inotify` instance of a server, and which tail each watch wakes. Dropping it stops its
/// task.
pub(crate) struct Hub {
    inner: Arc<Inner>,
    stop: tokio_util::sync::CancellationToken,
}

impl Hub {
    /// Must be called inside a runtime.
    pub fn new() -> io::Result<Arc<Self>> {
        let inotify = Inotify::init(InitFlags::IN_NONBLOCK | InitFlags::IN_CLOEXEC)
            .map_err(io::Error::from)?;
        // `AsyncFd::new` is deprecated because it takes any `AsRawFd`, which cannot promise the fd
        // stays open; `Fd` owns the `Inotify`, which closes it only when `Inner` drops. (`register`,
        // its replacement, is `unsafe`.)
        #[allow(deprecated)]
        let fd = AsyncFd::new(Fd(inotify))?;
        let inner = Arc::new(Inner {
            fd,
            wakes: Mutex::new(HashMap::new()),
        });
        let stop = tokio_util::sync::CancellationToken::new();
        let (task, cancel) = (inner.clone(), stop.clone());
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = cancel.cancelled() => return,
                    ready = task.fd.readable() => {
                        let Ok(mut guard) = ready else { return };
                        task.dispatch();
                        guard.clear_ready();
                    }
                }
            }
        });
        Ok(Arc::new(Self { inner, stop }))
    }

    /// Wake `wake` whenever `path` is written.
    fn watch(&self, path: &Path, wake: &Arc<Notify>) -> io::Result<WatchDescriptor> {
        let wd = self
            .inner
            .fd
            .get_ref()
            .0
            .add_watch(path, AddWatchFlags::IN_MODIFY)
            .map_err(io::Error::from)?;
        locked(&self.inner.wakes).insert(wd, wake.clone());
        Ok(wd)
    }

    fn unwatch(&self, wd: WatchDescriptor) {
        locked(&self.inner.wakes).remove(&wd);
        let _ = self.inner.fd.get_ref().0.rm_watch(wd);
    }
}

impl Drop for Hub {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}
