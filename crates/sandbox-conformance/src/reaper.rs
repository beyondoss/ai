//! Every process the harness starts is accounted for, including when the harness itself is killed.
//!
//! A child is its own process-group leader. Dropping its [`Leader`] kills the group (the same
//! `GroupKillGuard` the sandbox crate's test transports use). That does not run when the harness dies
//! by SIGKILL, so every leader and every container is also written to a small registry file, and one
//! watchdog process, started on first use in a group of its own, waits for the harness to be gone
//! and then kills the registered groups and removes the registered containers. The registry holds
//! only what is running: [`Leader`]'s drop and [`forget_container`] take their entries out, so a
//! recycled pid is never signalled.
use std::collections::BTreeSet;
use std::os::unix::process::CommandExt as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

use beyond_sandbox::procgroup::{GroupId, GroupKillGuard};

struct Registry {
    file: PathBuf,
    entries: BTreeSet<String>,
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();

/// Polls for the harness's pid to vanish (or turn zombie), then reads `$2` as lines `g PGID` /
/// `c NAME` and kills / removes them.
const WATCHDOG: &str = r#"
while [ -d "/proc/$1" ] && ! grep -q '^State:.*Z' "/proc/$1/status" 2>/dev/null; do sleep 0.3; done
if [ -f "$2" ]; then
  while read -r k v; do
    case "$k" in
      g) kill -s KILL -- "-$v" 2>/dev/null ;;
      c) docker rm -f "$v" >/dev/null 2>&1 ;;
    esac
  done < "$2"
  rm -f "$2"
fi
"#;

fn registry() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(|| {
        let file =
            std::env::temp_dir().join(format!("sandbox-harness-{}.reap", std::process::id()));
        let _ = std::fs::write(&file, "");
        let mut dog = Command::new("sh");
        dog.arg("-c")
            .arg(WATCHDOG)
            .arg("sh")
            .arg(std::process::id().to_string())
            .arg(&file)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0);
        // Detached on purpose: it outlives the harness by the moment it needs to clean up.
        let _ = dog.spawn();
        Mutex::new(Registry {
            file,
            entries: BTreeSet::new(),
        })
    })
}

fn update(f: impl FnOnce(&mut BTreeSet<String>)) {
    let mut r = registry().lock().unwrap_or_else(|e| e.into_inner());
    f(&mut r.entries);
    let text: String = r.entries.iter().map(|e| format!("{e}\n")).collect();
    // Written whole, then renamed over: the watchdog never reads half a file.
    let tmp = r.file.with_extension("tmp");
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, &r.file);
    }
}

/// A started process-group leader. Dropping it kills the group and takes it off the registry.
pub struct Leader {
    pgid: u32,
    _kill: GroupKillGuard,
}

impl Leader {
    /// For a child spawned with `.process_group(0)`, right after spawning it.
    pub fn new(child: &tokio::process::Child) -> Option<Leader> {
        let pid = child.id()?;
        update(|e| {
            e.insert(format!("g {pid}"));
        });
        Some(Leader {
            pgid: pid,
            _kill: GroupKillGuard {
                group: Some(GroupId::leader(pid)),
            },
        })
    }
}

impl Drop for Leader {
    fn drop(&mut self) {
        // The kill runs when `_kill` drops, just after; the entry goes first so a crash between the
        // two can only leave a group that is already empty.
        let pgid = self.pgid;
        update(|e| {
            e.remove(&format!("g {pgid}"));
        });
    }
}

/// Register a container by name before it is created.
pub fn watch_container(name: &str) {
    update(|e| {
        e.insert(format!("c {name}"));
    });
}

/// The container is gone.
pub fn forget_container(name: &str) {
    update(|e| {
        e.remove(&format!("c {name}"));
    });
}
