//! The `ai.usage` file sink: billing rows in hourly `.jsonl` files (`AI_USAGE_DIR`).
//!
//! Same contract as the stdout writer it replaces: a row is **written to the OS before the request's
//! accounting is done**, synchronously on the thread that finished the request, and is **never
//! dropped**. The bytes are the fmt layer's, unchanged — only the destination differs.
//!
//! # Layout
//!
//! `<dir>/ai-usage-YYYYMMDDTHH.jsonl`, the hour in UTC. One JSON object per line. The `.jsonl`
//! suffix is what logfwd's `*.jsonl` glob matches. A file is only ever appended to: a restart in the
//! same hour reopens it with `O_APPEND` (never truncates), and if the previous process died mid-row
//! the new one first ends that partial line so the next row does not fuse onto it.
//!
//! # Guarantees
//!
//! * **One `write(2)` per row**, under a mutex, with the file chosen under that same mutex. A row is
//!   never split across files or interleaved with another, and the hour boundary can only fall
//!   between two rows.
//! * **Back-pressure, not loss.** A failed write (disk full, directory gone, file replaced) is
//!   counted on `ai_usage_write_errors_total`, reported on stderr, and **retried every 100 ms until
//!   it lands** — the request thread waits, exactly as it would on a full stdout pipe. A partial
//!   write is truncated away before the retry, so a row never lands twice or in pieces.
//! * **fsync policy.** `write(2)` hands the row to the kernel: it survives a gateway crash or
//!   SIGKILL, which is the failure that matters on an ECS task. It does not survive power loss or a
//!   kernel panic until the page cache is flushed, so a ticker thread `fdatasync`s a dirty file every
//!   [`SYNC_EVERY`] (1 s), and rotation syncs the finished file before moving on. A host crash can
//!   therefore lose up to ~1 s of rows; an fsync per row would put a disk flush (hundreds of µs to
//!   ms) on every request for a failure the platform does not have. The ticker never touches the
//!   request path.
//! * **Retention.** logfwd tails and never rotates, so on each rotation (and at boot) files whose hour
//!   is more than [`RETAIN_HOURS`] (48) old are deleted. The current and previous hours are never
//!   candidates. Age is read from the file name, not mtime, so a restore or copy cannot make a
//!   current file look old.
//! * **One writer per file.** The hourly file is `flock`ed for as long as it is the active file. A
//!   process that finds it held (an overlapping old/new gateway on one host) writes
//!   `ai-usage-YYYYMMDDTHH.<pid>.jsonl` instead; still matched by the glob, still swept by retention.

use prometheus::IntCounter;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing_subscriber::fmt::MakeWriter;

/// How often a dirty file is `fdatasync`ed.
pub const SYNC_EVERY: Duration = Duration::from_secs(1);
/// Files whose hour is more than this many hours behind the current one are deleted.
pub const RETAIN_HOURS: u64 = 48;
/// Wait between attempts to land a row that failed to write.
const RETRY_EVERY: Duration = Duration::from_millis(100);

const PREFIX: &str = "ai-usage-";
const SUFFIX: &str = ".jsonl";

/// Source of "now"; injectable so tests can cross an hour boundary.
pub type Clock = Box<dyn Fn() -> SystemTime + Send + Sync>;

struct Active {
    /// Hours since the Unix epoch this file is for.
    hour: u64,
    file: Arc<File>,
    /// Bytes in the file, to truncate a partially written row away.
    len: u64,
}

struct Inner {
    dir: PathBuf,
    clock: Clock,
    errors: IntCounter,
    active: Option<Active>,
    /// Set by every write; cleared by the ticker, which syncs what it cleared.
    dirty: Arc<AtomicBool>,
}

/// The shared handle. Cloning is an `Arc` bump; install it as the usage layer's `MakeWriter`.
#[derive(Clone)]
pub struct UsageFiles(Arc<Mutex<Inner>>);

impl UsageFiles {
    /// Create `dir`, open the current hour's file and sweep old files. Failing here (unwritable
    /// directory) is a boot failure: a gateway that cannot record billing must not serve.
    pub fn open(dir: &Path, errors: IntCounter) -> io::Result<Self> {
        Self::open_with(dir, errors, Box::new(SystemTime::now), Some(SYNC_EVERY))
    }

    /// [`Self::open`] with an explicit clock and sync interval (`None`: no ticker).
    pub fn open_with(
        dir: &Path,
        errors: IntCounter,
        clock: Clock,
        sync_every: Option<Duration>,
    ) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut inner = Inner {
            dir: dir.to_path_buf(),
            clock,
            errors,
            active: None,
            dirty: Arc::new(AtomicBool::new(false)),
        };
        inner.ensure_current()?;
        let shared = Arc::new(Mutex::new(inner));
        if let Some(every) = sync_every {
            let weak = Arc::downgrade(&shared);
            std::thread::Builder::new()
                .name("ai-usage-sync".into())
                .spawn(move || sync_loop(&weak, every))?;
        }
        Ok(Self(shared))
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Land one whole row (including its trailing newline), blocking until it is written.
    pub fn write_row(&self, row: &[u8]) {
        self.lock().write_row(row);
    }

    /// Sync the active file now (shutdown, tests).
    pub fn sync(&self) {
        let (file, dirty) = {
            let g = self.lock();
            (g.active.as_ref().map(|a| a.file.clone()), g.dirty.clone())
        };
        if let Some(f) = file {
            dirty.store(false, Ordering::Relaxed);
            let _ = f.sync_data();
        }
    }
}

fn sync_loop(inner: &Weak<Mutex<Inner>>, every: Duration) {
    loop {
        std::thread::sleep(every);
        let Some(inner) = inner.upgrade() else { return };
        UsageFiles(inner).sync();
    }
}

impl Inner {
    fn hour_now(&self) -> u64 {
        (self.clock)()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() / 3600)
    }

    /// Make `active` the file for the current hour, rotating if the hour moved.
    fn ensure_current(&mut self) -> io::Result<()> {
        let hour = self.hour_now();
        if self.active.as_ref().is_some_and(|a| a.hour == hour) {
            return Ok(());
        }
        let next = open_hour(&self.dir, hour)?;
        if let Some(old) = self.active.replace(next) {
            // The finished file is complete: make it durable before moving on.
            let _ = old.file.sync_data();
        }
        self.dirty.store(false, Ordering::Relaxed);
        sweep(&self.dir, hour);
        Ok(())
    }

    fn write_row(&mut self, row: &[u8]) {
        let mut reported = false;
        loop {
            let err = match self.try_write(row) {
                Ok(()) => {
                    if reported {
                        let _ =
                            writeln!(io::stderr(), "ai.usage billing rows: file writes recovered");
                    }
                    return;
                }
                Err(e) => e,
            };
            if !reported {
                reported = true;
                self.errors.inc();
                let _ = writeln!(
                    io::stderr(),
                    "ai.usage billing row not yet written, retrying until it is: {err}"
                );
            }
            std::thread::sleep(RETRY_EVERY);
        }
    }

    fn try_write(&mut self, row: &[u8]) -> io::Result<()> {
        self.ensure_current()?;
        let Some(a) = self.active.as_mut() else {
            return Err(io::Error::other("no active usage file"));
        };
        match (&*a.file).write_all(row) {
            Ok(()) => {
                a.len += row.len() as u64;
                self.dirty.store(true, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                // Drop any partial row so the retry cannot duplicate or split it.
                let _ = a.file.set_len(a.len);
                Err(e)
            }
        }
    }
}

/// `ai-usage-YYYYMMDDTHH` for an hour count.
fn stem(hour: u64) -> String {
    let (y, m, d) = civil_from_days(hour / 24);
    format!("{PREFIX}{y:04}{m:02}{d:02}T{:02}", hour % 24)
}

/// Open (append, create) the file for `hour`, locked. An already-locked hourly file means another
/// process owns it: take a pid-suffixed name instead.
fn open_hour(dir: &Path, hour: u64) -> io::Result<Active> {
    let plain = dir.join(format!("{}{SUFFIX}", stem(hour)));
    let file = match open_locked(&plain)? {
        Some(f) => f,
        None => {
            let alt = dir.join(format!("{}.{}{SUFFIX}", stem(hour), std::process::id()));
            match open_locked(&alt)? {
                Some(f) => f,
                None => return Err(io::Error::other(format!("{} is locked", alt.display()))),
            }
        }
    };
    let mut len = file.metadata()?.len();
    if len > 0 {
        // A predecessor killed mid-row left an unterminated line: end it so our first row stands alone.
        let mut last = [0u8; 1];
        file.read_exact_at(&mut last, len - 1)?;
        if last[0] != b'\n' {
            (&file).write_all(b"\n")?;
            len += 1;
        }
    }
    Ok(Active {
        hour,
        file: Arc::new(file),
        len,
    })
}

fn open_locked(path: &Path) -> io::Result<Option<File>> {
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

/// Delete usage files more than [`RETAIN_HOURS`] behind `now_hour`. The current and previous hours
/// are never touched whatever the constant says. Best effort: a failed delete is reported and
/// retried at the next rotation.
fn sweep(dir: &Path, now_hour: u64) {
    let keep_from = now_hour
        .saturating_sub(RETAIN_HOURS)
        .min(now_hour.saturating_sub(1));
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let Some(hour) = name.to_str().and_then(parse_hour) else {
            continue;
        };
        if hour < keep_from
            && let Err(e) = std::fs::remove_file(entry.path())
            && e.kind() != io::ErrorKind::NotFound
        {
            let _ = writeln!(
                io::stderr(),
                "ai.usage retention: cannot delete {name:?}: {e}"
            );
        }
    }
}

/// The hour count of `ai-usage-YYYYMMDDTHH[.pid].jsonl`; `None` for any other name.
fn parse_hour(name: &str) -> Option<u64> {
    let rest = name.strip_prefix(PREFIX)?.strip_suffix(SUFFIX)?;
    let (ts, tail) = rest.split_at_checked(11)?;
    if !(tail.is_empty()
        || (tail.starts_with('.') && tail[1..].bytes().all(|b| b.is_ascii_digit())))
    {
        return None;
    }
    let b = ts.as_bytes();
    if b[8] != b'T'
        || !ts[..8].bytes().all(|c| c.is_ascii_digit())
        || !ts[9..].bytes().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let y: i64 = ts[0..4].parse().ok()?;
    let m: u32 = ts[4..6].parse().ok()?;
    let d: u32 = ts[6..8].parse().ok()?;
    let h: u64 = ts[9..11].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || h > 23 {
        return None;
    }
    let days = u64::try_from(days_from_civil(y, m, d)).ok()?;
    Some(days * 24 + h)
}

// Howard Hinnant's civil-date algorithms (days since 1970-01-01 <-> y/m/d); no date crate needed.
fn civil_from_days(z: u64) -> (i64, u32, u32) {
    let z = z as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = i64::from(if m > 2 { m - 3 } else { m + 9 });
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// One formatted event, collected whole and landed on drop (the fmt layer writes a row in several
/// `write` calls; one `write(2)` per row keeps rows unsplit).
pub struct UsageRow<'a> {
    buf: Vec<u8>,
    files: &'a UsageFiles,
}

impl Write for UsageRow<'_> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for UsageRow<'_> {
    fn drop(&mut self) {
        if !self.buf.is_empty() {
            self.files.write_row(&self.buf);
        }
    }
}

impl<'a> MakeWriter<'a> for UsageFiles {
    type Writer = UsageRow<'a>;
    fn make_writer(&'a self) -> Self::Writer {
        UsageRow {
            buf: Vec::with_capacity(1024),
            files: self,
        }
    }
}

#[cfg(test)]
mod tests;
