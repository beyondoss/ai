//! What the clients were told, written down as it happens.
//!
//! The checker's job is to compare promises against the filesystem, so the promises have to be
//! recorded at the moment they are made rather than reconstructed afterwards. An acknowledged line is
//! a promise: if a client was told a turn committed, that turn must survive every subsequent
//! takeover, drain and crash.
//!
//! One file, one JSON object per line, a monotonic sequence number. A wall clock is recorded too, but
//! nothing is ever *ordered* by it — under chaos the interesting events are milliseconds apart and
//! the sequence is what makes them comparable.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

pub struct History {
    path: PathBuf,
    file: Mutex<std::fs::File>,
    seq: AtomicU64,
}

impl History {
    pub fn create(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::create(path).map_err(|e| format!("history {path:?}: {e}"))?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            seq: AtomicU64::new(0),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Record one event. Failures are ignored on purpose: a simulator that dies because it could not
    /// write its own log has destroyed the run it was observing.
    pub fn record(&self, kind: &str, detail: Value) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let line = json!({ "seq": seq, "at_ms": at_ms, "kind": kind, "detail": detail });
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }

    /// Read a recorded history back, for the checker.
    pub fn read(path: &Path) -> Result<Vec<Value>, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("history {path:?}: {e}"))?;
        Ok(text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect())
    }
}
