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
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write as _};
use std::os::fd::{AsFd, AsRawFd, RawFd};

use crate::spool::Base;
use std::os::unix::fs::{FileExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use nix::fcntl::{FallocateFlags, OFlag, fallocate, openat};
use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify, WatchDescriptor};
use nix::sys::stat::Mode;
use nix::unistd::{Whence, lseek};
use tokio::io::unix::AsyncFd;
use tokio::sync::Notify;

use crate::proc::locked;

/// Bytes read at once, and the step in which read bytes are returned to the filesystem.
const STEP: usize = 64 * 1024;

/// One job's spool directory, named relative to its held base (`spool::Base`): its files are made
/// through a descriptor of the directory itself, and every later open goes through the base's
/// descriptor (`/proc/self/fd/<base>/<mark>`), so no name swapped in above the base is followed.
/// One descriptor serves every job of a base: an ended job holds none.
#[derive(Debug)]
pub(crate) struct Spool {
    /// The name the ledger records.
    dir: PathBuf,
    base: Base,
    mark: String,
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
    /// append mode, every step relative to a descriptor, `O_NOFOLLOW`, `O_EXCL`.
    pub fn create(base: &Base, mark: &str) -> io::Result<(Self, Writers)> {
        let fd = base.make_dir(mark)?;
        let spool = Self {
            dir: base.path().join(mark),
            base: base.clone(),
            mark: mark.to_string(),
        };
        let open = |name: &str| {
            openat(
                &fd,
                name,
                OFlag::O_WRONLY
                    | OFlag::O_APPEND
                    | OFlag::O_CREAT
                    | OFlag::O_EXCL
                    | OFlag::O_NOFOLLOW
                    | OFlag::O_CLOEXEC,
                Mode::from_bits_truncate(0o600),
            )
            .map(File::from)
            .map_err(io::Error::from)
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

    /// The spool at `dir`, which a previous execd made (the ledger names it), if it is in `base`
    /// and is still there as a real directory.
    pub fn at(base: &Base, dir: &Path) -> Option<Self> {
        if dir.parent()? != base.path() {
            return None;
        }
        let mark = dir.file_name()?.to_str()?;
        // A real directory, never a link.
        drop(base.open_dir(mark).ok()?);
        Some(Self {
            dir: dir.to_path_buf(),
            base: base.clone(),
            mark: mark.to_string(),
        })
    }

    /// The directory, which the ledger records.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// A path to this job's directory through the base's descriptor: what every open of a file in
    /// the spool goes through.
    pub fn access(&self) -> PathBuf {
        self.base.access().join(&self.mark)
    }

    /// The base's descriptor, to be kept by everything that stores a path made by
    /// [`Self::access`]: its number stays the base's for as long as one of them is alive.
    pub fn guard(&self) -> Arc<std::os::fd::OwnedFd> {
        self.base.guard()
    }

    /// Where this job's ring keeps its chunk files: with the spool, so a restart finds them.
    pub fn ring_dir(&self) -> PathBuf {
        self.access().join("ring")
    }

    /// Delete the directory and the files in it. Idempotent.
    pub fn remove(&self) {
        self.base.remove_dir_all(&self.mark);
    }

    /// Delete every spool under `base` that is not one of `keep`: the ones no job in the ledger
    /// names (a job that ended or was removed before the restart). Only real directories, by name
    /// relative to the held directory.
    pub fn retain(base: &Base, keep: &[&Path]) {
        for name in base.dirs() {
            if !keep.contains(&base.path().join(&name).as_path()) {
                base.remove_dir_all(&name);
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
            Stream::open(&spool.access().join("out"), start, 0)?,
            Stream::open(&spool.access().join("err"), start, 1)?,
        ];
        let wake = Arc::new(Notify::new());
        let wds = [
            hub.watch(&spool.access().join("out"), &wake)?,
            hub.watch(&spool.access().join("err"), &wake)?,
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

    /// Read both files, a buffer from each in turn so neither starves the other, and hand each
    /// piece to `sink` (`stderr` for the `err` file). `sink` returns `false` to stop (the job is
    /// gone); then so does this. At most `rounds` rounds per call, so a job writing flat out gives
    /// the caller a turn between batches. Blocking: the sink writes the ring's chunks and journal,
    /// and this punches the files.
    pub fn drain_batch(
        &mut self,
        rounds: usize,
        mut sink: impl FnMut(&[u8], bool) -> bool,
    ) -> Drained {
        for _ in 0..rounds {
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
                    return Drained::Stopped;
                }
                s.punch(&mut self.warned);
            }
            if !any {
                return Drained::Idle;
            }
        }
        Drained::More
    }
}

/// How [`Tail::drain_batch`] ended. `Stopped` is also what a panicking batch counts as.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum Drained {
    /// The sink refused a piece: the job is gone.
    #[default]
    Stopped,
    /// Both files are read to their ends.
    Idle,
    /// The batch used its rounds up; there may be more.
    More,
}

/// Rounds of reading per [`Tail::drain_batch`] call off the runtime: at most 2 x 16 x 64 KiB.
pub const ROUNDS: usize = 16;

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
    /// Keeps the descriptor `dir` names open (see [`Spool::guard`]).
    _hold: Arc<std::os::fd::OwnedFd>,
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
        hold: Arc<std::os::fd::OwnedFd>,
        start: [u64; 2],
        pieces: Vec<(bool, Vec<u8>)>,
        ring_tail: usize,
    ) -> io::Result<Self> {
        let held = pieces.iter().map(|p| p.1.len()).sum();
        let mut j = Self {
            dir: dir.to_path_buf(),
            _hold: hold,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod hold_tests {
    use super::*;
    use crate::ring::Ring;
    use crate::spool::JobsDir;

    fn stable() -> (tempfile::TempDir, JobsDir) {
        let base = tempfile::tempdir().unwrap();
        let jobs = JobsDir::open(Some(base.path()), Some(Path::new("/state"))).unwrap();
        (base, jobs)
    }

    fn names(d: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(d)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    /// A job's ring and journal keep the base's descriptor open, so after the server that made it is
    /// gone its number is not another server's: a late flush fails in the removed directory, never
    /// writes into a later server's spool of the same name. Red when the descriptor is not held.
    #[test]
    fn a_stale_ring_and_journal_never_write_into_a_later_servers_spool() {
        let (_base1, jobs1) = stable();
        let (a, _w) = Spool::create(jobs1.stable(), "a").unwrap();
        let mut journal = Journal::create(&a.access(), a.guard(), [0, 0], Vec::new(), 0).unwrap();
        let mut ring = Ring::new(1 << 20, a.ring_dir()).holding(a.guard()).kept();
        a.remove();
        drop(a);
        drop(jobs1);
        // A later server: with the numbers free it is given the first one's.
        let (_base2, jobs2) = stable();
        let (b, _wb) = Spool::create(jobs2.stable(), "a").unwrap();
        let _b_journal = Journal::create(&b.access(), b.guard(), [0, 0], Vec::new(), 0).unwrap();
        let before = names(&b.access());
        let journal_before = fs::read(b.access().join("journal")).unwrap();
        // The first one's flushes: a journal rewrite and a full chunk of the ring.
        journal.record(false, b"x", 0);
        ring.push(&vec![b'z'; 70_000], false);
        assert_eq!(
            names(&b.access()),
            before,
            "a stale writer reached a later spool"
        );
        assert_eq!(
            fs::read(b.access().join("journal")).unwrap(),
            journal_before
        );
    }

    /// Every path under `d`, with each file's bytes: what a stale writer must leave as it was.
    fn tree(d: &Path) -> Vec<(PathBuf, Option<Vec<u8>>)> {
        let mut out = Vec::new();
        let mut stack = vec![d.to_path_buf()];
        while let Some(p) = stack.pop() {
            for e in fs::read_dir(&p).unwrap().flatten() {
                if e.file_type().unwrap().is_dir() {
                    out.push((e.path(), None));
                    stack.push(e.path());
                } else {
                    out.push((e.path(), Some(fs::read(e.path()).unwrap())));
                }
            }
        }
        out.sort();
        out
    }

    /// The same server: job a is removed and job b made under the same base. a's stale ring and
    /// journal neither touch b nor bring a's directory back, and nothing under the base gains a file.
    /// Red when a stale writer can create its directory or reach b.
    #[test]
    fn a_removed_jobs_stale_ring_and_journal_touch_nothing_under_the_base() {
        let (_base, jobs) = stable();
        let (a, _w) = Spool::create(jobs.stable(), "a").unwrap();
        let mut journal = Journal::create(&a.access(), a.guard(), [0, 0], Vec::new(), 0).unwrap();
        let mut ring = Ring::new(1 << 20, a.ring_dir()).holding(a.guard()).kept();
        a.remove();
        drop(a);
        let (b, _wb) = Spool::create(jobs.stable(), "b").unwrap();
        let _b_journal = Journal::create(&b.access(), b.guard(), [0, 0], Vec::new(), 0).unwrap();
        let b_listing = names(&b.access());
        let b_journal = fs::read(b.access().join("journal")).unwrap();
        let before = tree(jobs.stable().path());
        journal.record(false, b"x", 0);
        ring.push(&vec![b'z'; 70_000], false);
        assert_eq!(
            names(&b.access()),
            b_listing,
            "a stale writer reached job b"
        );
        assert_eq!(fs::read(b.access().join("journal")).unwrap(), b_journal);
        assert!(
            !jobs.stable().path().join("a").exists(),
            "a's directory came back"
        );
        assert_eq!(tree(jobs.stable().path()), before, "the base changed");
    }

    /// `retain` and a job's removal go by name relative to the held directory and remove only real
    /// directories: a planted link, and the directory it names, are never touched. Red against a
    /// retain that follows links.
    #[test]
    fn retain_and_removal_never_follow_a_planted_link() {
        let (base, jobs) = stable();
        let victim = tempfile::tempdir().unwrap();
        fs::create_dir(victim.path().join("keep")).unwrap();
        fs::write(victim.path().join("keep/f"), b"precious").unwrap();
        let _ = &base;
        let root = jobs.stable().path().to_path_buf();
        std::os::unix::fs::symlink(victim.path(), root.join("planted")).unwrap();
        let (gone, _w) = Spool::create(jobs.stable(), "gone").unwrap();
        // A job's directory name replaced by a link, then the job removed.
        let (swapped, _w2) = Spool::create(jobs.stable(), "swapped").unwrap();
        fs::rename(root.join("swapped"), root.join("swapped.moved")).unwrap();
        std::os::unix::fs::symlink(victim.path(), root.join("swapped")).unwrap();
        swapped.remove();
        Spool::retain(jobs.stable(), &[]);
        assert!(!root.join("gone").exists());
        drop(gone);
        assert_eq!(
            fs::read(victim.path().join("keep/f")).unwrap(),
            b"precious",
            "a link's target was touched"
        );
        assert!(
            fs::symlink_metadata(root.join("planted"))
                .unwrap()
                .is_symlink()
        );
    }
}
