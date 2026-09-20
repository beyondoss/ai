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

/// A prepared substrate: the shard directories, and whatever has to be torn down afterwards.
pub struct Substrate {
    pub kind: Kind,
    /// Shard name → the path replicas are given with `--shard <name>=<path>`.
    pub shards: Vec<(String, PathBuf)>,
    /// Kept so the temporary root outlives the run.
    _root: tempfile::TempDir,
}

impl Substrate {
    /// Prepare `shards` named `s1..sN`.
    pub fn prepare(kind: Kind, shards: usize) -> Result<Self, String> {
        if kind == Kind::Nfs {
            // Deliberately a clear refusal rather than a half-configured export. Standing one up
            // means installing `nfs-kernel-server`, starting a daemon and opening ports on the host
            // — a change to the machine, not to this repo, and not one a test harness should make on
            // its own initiative.
            return Err(
                "the `nfs` substrate needs `nfs-kernel-server` and `nfs-common` installed and root \
                 to export and mount. Install them, then re-run with `--substrate nfs`; until then \
                 `--substrate local-dir` runs every claim that does not depend on a shared mount."
                    .to_string(),
            );
        }
        let root = tempfile::tempdir().map_err(|e| format!("substrate root: {e}"))?;
        let mut out = Vec::new();
        for i in 1..=shards {
            let name = format!("s{i}");
            let path = root.path().join(&name);
            std::fs::create_dir_all(&path).map_err(|e| format!("shard {name}: {e}"))?;
            out.push((name, path));
        }
        Ok(Self {
            kind,
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
