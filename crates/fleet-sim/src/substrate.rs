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
    sudo(&["true"]).map_err(|e| format!("the `nfs` substrate needs passwordless sudo: {e}"))
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
        for m in mounts.iter().rev() {
            let _ = sudo(&["umount", "-l", &m.display().to_string()]);
        }
        let _ = sudo(&["exportfs", "-u", &format!("127.0.0.1:{}", export.display())]);
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

        // `no_root_squash` so the replicas (running as this user) own what they write, and
        // `no_subtree_check` because the export is a temporary directory rather than a real
        // filesystem. `fsid` is required for an NFSv4 export of a non-device directory.
        sudo(&[
            "exportfs",
            "-o",
            "rw,sync,no_subtree_check,no_root_squash,fsid=8421",
            &format!("127.0.0.1:{}", export.display()),
        ])?;

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
