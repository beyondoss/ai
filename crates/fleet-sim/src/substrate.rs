//! Where a replica set's shards live.
//!
//! The design's correctness claims are about a **shared network filesystem**: `O_APPEND` is not
//! atomic across clients, which is why an owner writes only its own epoch segment; and a dead
//! client's advisory lock is released when its *lease* lapses, not when its process dies, which is
//! why a takeover can be slow but never unsafe. Neither behaviour exists on a local filesystem.
//!
//! So this is two substrates, and being honest about the difference is the point:
//!
//! - [`Substrate::LocalDir`] is a plain directory. Everything above the filesystem is real — real
//!   replica processes, real locks, real epoch segments, real grants, real chaos. What it cannot
//!   show you is any behaviour that only appears when two machines share a mount.
//! - [`Substrate::Nfs`] exports that directory over NFSv4.1 and mounts it once per replica, each in
//!   its own mount namespace so a frozen export degrades one replica rather than all of them.
//!
//! Which claims each can actually settle is spelled out in `scenarios.rs`. A scenario that needs the
//! network filesystem says so and skips rather than passing vacuously on a local directory, because a
//! green result that proved nothing is worse than a skip that says why.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// How the shards under test are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A plain directory. No privileges, fast, and blind to every cross-client property.
    LocalDir,
    /// A real NFSv4.1 export, mounted per replica. Needs root and `nfs-kernel-server`.
    Nfs,
}

impl Kind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "local-dir" => Ok(Self::LocalDir),
            "nfs" => Ok(Self::Nfs),
            other => Err(format!(
                "unknown substrate {other:?} (expected `local-dir` or `nfs`)"
            )),
        }
    }

    /// Does this substrate exhibit the cross-client behaviour the fencing design is about?
    ///
    /// Used by scenarios to skip rather than pass vacuously. A local directory has one client by
    /// construction: its `O_APPEND` *is* atomic, and a lock dies with the process holding it, so a
    /// scenario written to catch a lost write or a lapsed lease would go green having exercised
    /// nothing.
    pub fn is_shared_filesystem(self) -> bool {
        matches!(self, Self::Nfs)
    }
}

/// Run a command under `sudo -n`, failing with what it said.
fn sudo(args: &[&str]) -> Result<(), String> {
    let out = std::process::Command::new("sudo")
        .arg("-n")
        .args(args)
        .output()
        .map_err(|e| format!("sudo {}: {e}", args.join(" ")))?;
    if out.status.success() {
        return Ok(());
    }
    Err(format!(
        "sudo {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr).trim()
    ))
}

/// The name of the marker every export this simulator creates carries, holding the pid that made
/// it. The sweep uses it to tell its own debris from an export somebody else on this host owns —
/// without it, "clean up stale NFS exports" is a command that unexports production.
const MARKER: &str = ".fleet-sim-export";

/// A distinct `fsid` per export.
///
/// This was `fsid=8421`, a constant, and that was a real bug rather than a cosmetic one. NFSv4
/// presents a single namespace keyed by `fsid`, so two exports claiming the same one are not two
/// exports — the server resolves a mount to whichever it finds, which after a killed run means
/// mounting a deleted directory and hanging on the first operation. A constant also made two
/// fleet-sims on one host mutually destructive.
fn unique_fsid() -> u32 {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    // Non-zero: 0 is the pseudo-root on some server configurations.
    (pid.wrapping_mul(2_654_435_761).wrapping_add(nanos) % 0xFFFF_FF00) + 1
}

/// Exports and mounts that must come down even if this process is killed.
///
/// `Drop` does not run on `SIGTERM`, which is exactly how a `timeout`-ed run leaves a mount behind.
/// The registry is what the signal handler tears down, and it is the same list `Drop` uses, so the
/// two paths cannot disagree about what was created.
static PENDING: Mutex<Vec<(PathBuf, Vec<PathBuf>)>> = Mutex::new(Vec::new());

/// Tear down one export and its mounts. Best-effort and idempotent in both directions: unmounting
/// something already gone and unexporting something already withdrawn are both fine.
fn teardown(export: &Path, mounts: &[PathBuf]) {
    for m in mounts.iter().rev() {
        let _ = sudo(&["umount", "-l", &m.display().to_string()]);
    }
    let _ = sudo(&["exportfs", "-u", &format!("127.0.0.1:{}", export.display())]);
}

/// Bring down everything this process registered. Safe to call twice.
pub fn teardown_all() {
    let pending = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    for (export, mounts) in pending {
        teardown(&export, &mounts);
    }
}

/// Undo what a *previous* run was killed before undoing.
///
/// Only exports carrying [`MARKER`] with a pid that is gone, which is the whole safety argument: an
/// export without the marker was not made here and is not touched, and one whose pid is still alive
/// belongs to a fleet-sim running right now.
fn sweep_orphans() {
    let Ok(out) = std::process::Command::new("sudo")
        .arg("-n")
        .args(["exportfs", "-v"])
        .output()
    else {
        return;
    };
    let listing = String::from_utf8_lossy(&out.stdout).into_owned();
    let mounts = std::fs::read_to_string("/proc/mounts").unwrap_or_default();
    for line in listing.lines() {
        // An indented line is the client/options continuation of the path above it.
        if line.starts_with([' ', '\t']) || line.trim().is_empty() {
            continue;
        }
        let Some(path) = line.split_whitespace().next() else {
            continue;
        };
        let export = PathBuf::from(path);
        let Ok(pid) = std::fs::read_to_string(export.join(MARKER)) else {
            continue; // not ours
        };
        if std::path::Path::new(&format!("/proc/{}", pid.trim())).exists() {
            continue; // a live run owns it
        }
        let stale: Vec<PathBuf> = mounts
            .lines()
            .filter_map(|m| {
                let mut f = m.split_whitespace();
                let dev = f.next()?;
                let at = f.next()?;
                dev.starts_with(&format!("127.0.0.1:{path}"))
                    .then(|| PathBuf::from(at))
            })
            .collect();
        eprintln!(
            "fleet-sim: cleaning up an export left by pid {} ({path})",
            pid.trim()
        );
        teardown(&export, &stale);
    }
}

/// Say what is missing before touching anything, so a half-built export is never left behind.
fn preflight_nfs() -> Result<(), String> {
    let mut missing = Vec::new();
    for tool in ["/usr/sbin/exportfs", "/sbin/exportfs"] {
        if std::path::Path::new(tool).exists() {
            missing.clear();
            break;
        }
        missing.push("nfs-kernel-server");
    }
    if !std::path::Path::new("/sbin/mount.nfs4").exists()
        && !std::path::Path::new("/usr/sbin/mount.nfs4").exists()
    {
        missing.push("nfs-common");
    }
    if !missing.is_empty() {
        missing.sort_unstable();
        missing.dedup();
        return Err(format!(
            "the `nfs` substrate needs {} installed, and root to export and mount. Installing them starts a daemon and opens ports on this host — a change to the machine, not to this repo, so the simulator will not do it for you. Until then `--substrate local-dir` runs every claim that does not depend on a shared mount.",
            missing.join(" and ")
        ));
    }
    sudo(&["true"]).map_err(|e| format!("the `nfs` substrate needs passwordless sudo: {e}"))?;
    sweep_orphans();
    Ok(())
}

/// A prepared substrate: the shard directories, and whatever has to be torn down afterwards.
pub struct Substrate {
    pub kind: Kind,
    /// Shard name → the path replicas are given with `--shard <name>=<path>`.
    pub shards: Vec<(String, PathBuf)>,
    /// What the NFS substrate has to undo: the export path, and the mount points under it.
    nfs_teardown: Option<(PathBuf, Vec<PathBuf>)>,
    /// Kept so the temporary root outlives the run.
    _root: tempfile::TempDir,
}

impl Drop for Substrate {
    fn drop(&mut self) {
        // A simulator that leaves NFS mounts behind has damaged the machine it was borrowing. Every
        // step is best-effort and in reverse order: unmount each shard, then withdraw the export. A
        // lazy unmount (`-l`) because a replica that outlived its scenario may still hold a
        // descriptor, and a mount that will not come down is worse than one that comes down late.
        let Some((export, mounts)) = self.nfs_teardown.take() else {
            return;
        };
        if let Ok(mut pending) = PENDING.lock() {
            pending.retain(|(e, _)| e != &export);
        }
        teardown(&export, &mounts);
    }
}

impl Substrate {
    /// Prepare `shards` named `s1..sN`.
    pub fn prepare(kind: Kind, shards: usize) -> Result<Self, String> {
        if kind == Kind::Nfs {
            preflight_nfs()?;
            return Self::prepare_nfs(shards);
        }
        Self::prepare_local(shards)
    }

    fn prepare_local(shards: usize) -> Result<Self, String> {
        let root = tempfile::tempdir().map_err(|e| format!("substrate root: {e}"))?;
        let mut out = Vec::new();
        for i in 1..=shards {
            let name = format!("s{i}");
            let path = root.path().join(&name);
            std::fs::create_dir_all(&path).map_err(|e| format!("shard {name}: {e}"))?;
            out.push((name, path));
        }
        Ok(Self {
            kind: Kind::LocalDir,
            shards: out,
            nfs_teardown: None,
            _root: root,
        })
    }

    /// Export the shard directories over NFSv4.1 and mount them back, so every replica reaches its
    /// shards through a real NFS client.
    ///
    /// Loopback, deliberately: the point is the **protocol**, not the network. An NFS client talking
    /// to `127.0.0.1` still goes through the full client stack — close-to-open consistency, an
    /// `O_APPEND` that is not atomic across clients, and locks held on a lease rather than by a
    /// process — which is the entire set of behaviours a local directory cannot show.
    fn prepare_nfs(shards: usize) -> Result<Self, String> {
        let root = tempfile::tempdir().map_err(|e| format!("substrate root: {e}"))?;
        let export = root.path().join("export");
        let mnt = root.path().join("mnt");
        std::fs::create_dir_all(&export).map_err(|e| format!("export dir: {e}"))?;
        std::fs::create_dir_all(&mnt).map_err(|e| format!("mount dir: {e}"))?;

        let mut out = Vec::new();
        for i in 1..=shards {
            let name = format!("s{i}");
            std::fs::create_dir_all(export.join(&name))
                .map_err(|e| format!("shard {name}: {e}"))?;
            std::fs::create_dir_all(mnt.join(&name)).map_err(|e| format!("mnt {name}: {e}"))?;
            out.push((name, mnt.join(format!("s{i}"))));
        }

        // The marker goes down before the export does, so a kill between the two lines leaves a
        // marked directory and no export rather than an export the sweep cannot claim.
        std::fs::write(export.join(MARKER), std::process::id().to_string())
            .map_err(|e| format!("export marker: {e}"))?;

        // `no_root_squash` so the replicas (running as this user) own what they write, and
        // `no_subtree_check` because the export is a temporary directory rather than a real
        // filesystem. `fsid` is required for an NFSv4 export of a non-device directory.
        // `insecure` is not laxity, it is what makes the *client* options faithful. EFS's documented
        // mount includes `noresvport` — the client uses an ordinary high port rather than a reserved
        // one, which is how it survives a reconnect without exhausting the privileged range. A Linux
        // kernel export defaults to `secure`, which requires a reserved source port, so the two
        // together produce `EPERM` at mount time. Dropping `noresvport` would "fix" it by testing
        // mount options no replica will ever use; marking the export `insecure` keeps the client
        // exactly as EFS recommends. The export is bound to 127.0.0.1 regardless.
        sudo(&[
            "exportfs",
            "-o",
            &format!(
                "rw,sync,no_subtree_check,no_root_squash,insecure,fsid={}",
                unique_fsid()
            ),
            &format!("127.0.0.1:{}", export.display()),
        ])?;

        // Registered before the first mount: from here on a signal tears down whatever exists.
        if let Ok(mut pending) = PENDING.lock() {
            pending.push((export.clone(), out.iter().map(|(_, p)| p.clone()).collect()));
        }

        for (name, path) in &out {
            sudo(&[
                "mount",
                "-t",
                "nfs4",
                // EFS's own documented recommendation, so the client behaves the way a replica's
                // will in production rather than however this kernel's defaults happen to be set.
                "-o",
                "nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport",
                &format!("127.0.0.1:{}/{name}", export.display()),
                &path.display().to_string(),
            ])?;
        }

        Ok(Self {
            kind: Kind::Nfs,
            nfs_teardown: Some((export, out.iter().map(|(_, p)| p.clone()).collect())),
            shards: out,
            _root: root,
        })
    }

    /// The `--shard` arguments a replica is launched with.
    pub fn shard_args(&self) -> Vec<(&str, &Path)> {
        self.shards
            .iter()
            .map(|(n, p)| (n.as_str(), p.as_path()))
            .collect()
    }

    /// Every session directory currently on `shard`, for the checker.
    pub fn session_dirs(&self, shard: &str, tenant: &str) -> Vec<PathBuf> {
        let Some((_, root)) = self.shards.iter().find(|(n, _)| n == shard) else {
            return Vec::new();
        };
        let sessions = root.join(tenant).join("sessions");
        let Ok(entries) = std::fs::read_dir(sessions) else {
            return Vec::new();
        };
        entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect()
    }
}
