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
    /// A shared filesystem **somebody else provisioned**, already mounted, at paths the caller
    /// names. Creates nothing and tears nothing down.
    ///
    /// This is what runs the matrix against real EFS: the shards are an EFS filesystem mounted into
    /// every task by the platform, the replicas are ECS tasks the simulator attached to rather than
    /// spawned, and faults are performed by a caller-supplied command (see [`crate::faults`]).
    /// Nothing about EFS or ECS is known here — only that the paths exist and that two clients can
    /// reach them, which is the only property the storage claims actually rest on.
    Attached,
}

impl Kind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "local-dir" => Ok(Self::LocalDir),
            "nfs" => Ok(Self::Nfs),
            "attached" => Ok(Self::Attached),
            other => Err(format!(
                "unknown substrate {other:?} (expected `local-dir`, `nfs` or `attached`)"
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
        matches!(self, Self::Nfs | Self::Attached)
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
    // Non-zero because 0 is the pseudo-root on some server configurations, and inside `i32` because
    // `exportfs -v` prints the fsid signed — a value above 2^31 comes back as a negative number,
    // which is at best unreadable and at worst something a parser further along gets wrong.
    (pid.wrapping_mul(2_654_435_761).wrapping_add(nanos) % 0x7FFF_FF00) + 1
}

/// One replica's private NFS client: a network namespace, a veth pair to this host, and the
/// addresses either end holds.
///
/// **Why a namespace and not just another address.** The obvious cheap trick — mount each replica
/// from its own loopback address — does not work, and failing quietly is the worst part. A Linux
/// NFSv4 client identifies itself to the server by a string built from its UTS hostname, so every
/// mount made by one host is *one client* sharing *one lease*, whatever address it dialled. The
/// server confirms it: two mounts, one entry under `/proc/fs/nfsd/clients`. Cutting one of them off
/// therefore stops renewing the lease for both, and the peer that was supposed to take the session
/// over loses its own storage at the same moment. Measured: the successor never got the lock, and
/// the first reading of that was "the design strands sessions", which was wrong.
///
/// A network namespace plus a UTS namespace with its own hostname gives a genuinely separate client
/// — its own `clientid`, its own lease, its own revocation. The **mount namespace is deliberately
/// not** unshared, so the mount lands in this host's namespace where an ordinary replica process
/// reaches it; the RPC still travels through the namespace the mount was made in, which is the whole
/// point. Verified: three distinct clients, each named for its namespace, and a partitioned one's
/// lock released to its peer after 85 s against a 90 s lease.
struct NetClient {
    ns: String,
    host_if: String,
    client_if: String,
    host_ip: String,
}

/// Everything one prepared NFS substrate has to give back: the export, its mounts, and the network
/// namespaces and veth pairs the clients live in.
struct Undo {
    export: PathBuf,
    mounts: Vec<PathBuf>,
    nets: Vec<(String, String)>,
}

/// Exports and mounts that must come down even if this process is killed.
///
/// `Drop` does not run on `SIGTERM`, which is exactly how a `timeout`-ed run leaves a mount behind.
/// The registry is what the signal handler tears down, and it is the same list `Drop` uses, so the
/// two paths cannot disagree about what was created.
static PENDING: Mutex<Vec<Undo>> = Mutex::new(Vec::new());

/// Tear one substrate down. Best-effort, idempotent in every direction, and in reverse order:
/// unmount, delete the veth pairs (which takes both ends), delete the namespaces, withdraw the
/// export. Unmounting something already gone and unexporting something already withdrawn are both
/// fine, which is what lets this run from `Drop` and from a signal handler without coordinating.
fn teardown(undo: &Undo) {
    // **Every link comes up first**, including one a scenario deliberately took down. An NFS client
    // cannot finish unmounting without reaching its server: it still has opens to close and a lease
    // to surrender. Unmounting a partitioned client leaves that work outstanding forever.
    for (ns, host_if) in &undo.nets {
        let client_if = host_if.replacen("fh", "fc", 1);
        let _ = sudo(&[
            "ip", "netns", "exec", ns, "ip", "link", "set", &client_if, "up",
        ]);
    }

    // Then unmount **synchronously**, and only fall back to a lazy unmount if that fails.
    //
    // This was `umount -l` unconditionally, and that is what leaked. A lazy unmount detaches the
    // tree from the namespace and returns immediately, leaving the superblock — and the NFSv4
    // client's state manager — alive in the background, still needing the network. The next lines
    // then deleted the veth and the namespace out from under it, so the state manager was left
    // retrying against an address that no longer existed, in uninterruptible sleep, forever. A
    // kernel thread per client named `<server-ip>-manager`, each pinning +1.00 on the host's load
    // average until it reboots. Three of them accumulated before anyone looked at why an idle
    // machine showed a load of five.
    for m in undo.mounts.iter().rev() {
        let at = m.display().to_string();
        if sudo(&["umount", &at]).is_err() {
            // A mount that genuinely will not come down (a server that stopped answering mid-run) is
            // still better detached than left: `-f` gives the client permission to abandon its RPCs
            // rather than retry them, which is what makes the lazy detach safe to follow with a
            // network teardown.
            let _ = sudo(&["umount", "-f", "-l", &at]);
        }
    }

    for (ns, host_if) in &undo.nets {
        // Deleting one end of a veth deletes the pair, wherever the peer lives.
        let _ = sudo(&["ip", "link", "del", host_if]);
        let _ = sudo(&["ip", "netns", "del", ns]);
    }
    let _ = sudo(&[
        "exportfs",
        "-u",
        &format!("{EXPORT_NET}:{}", undo.export.display()),
    ]);
}

/// Bring down everything this process registered. Safe to call twice.
pub fn teardown_all() {
    let pending = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    for undo in &pending {
        teardown(undo);
    }
}

/// Who the export is offered to. A private range reached only over the veth pairs this simulator
/// creates — nothing off this host is routed here.
const EXPORT_NET: &str = "10.0.0.0/8";

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
        // An export whose directory no longer exists cannot be anyone's live export, and it has no
        // marker left to identify it by — which is exactly how four of these accumulated before this
        // arm existed: the run's temporary directory was removed, so the marker check below could
        // never claim them and the sweep skipped them forever.
        let vanished = !export.exists();
        if !vanished {
            let Ok(pid) = std::fs::read_to_string(export.join(MARKER)) else {
                continue; // not ours
            };
            if std::path::Path::new(&format!("/proc/{}", pid.trim())).exists() {
                continue; // a live run owns it
            }
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
        eprintln!("fleet-sim: cleaning up an export left behind ({path})");
        teardown(&Undo {
            export,
            mounts: stale,
            nets: Vec::new(),
        });
    }
    sweep_orphan_namespaces();
}

/// Delete network namespaces this simulator made whose owning run is gone.
///
/// Named for the pid that created them, so the test is the same one the export sweep uses: a
/// namespace belonging to a live process is left alone, and anything not named `fsim<pid>c<n>` is
/// never touched at all.
fn sweep_orphan_namespaces() {
    let Ok(out) = std::process::Command::new("sudo")
        .arg("-n")
        .args(["ip", "netns", "list"])
        .output()
    else {
        return;
    };
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some(name) = line.split_whitespace().next() else {
            continue;
        };
        let Some(rest) = name.strip_prefix("fsim") else {
            continue;
        };
        let Some((pid, _)) = rest.split_once('c') else {
            continue;
        };
        if pid.is_empty() || std::path::Path::new(&format!("/proc/{pid}")).exists() {
            continue;
        }
        eprintln!("fleet-sim: cleaning up a network namespace left behind ({name})");
        let _ = sudo(&["ip", "link", "del", &name.replacen("fsim", "fh", 1)]);
        let _ = sudo(&["ip", "netns", "del", name]);
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
    for tool in ["/usr/sbin/ip", "/usr/bin/nsenter", "/usr/bin/unshare"] {
        if !std::path::Path::new(tool).exists() {
            missing.push(if tool.ends_with("ip") {
                "iproute2"
            } else {
                "util-linux"
            });
        }
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
    /// The canonical view of each shard, for the checker: the **server side** of the export under
    /// NFS, the directory itself otherwise.
    ///
    /// Deliberately not one of the client mounts. A checker reading through a client is reading
    /// through that client's attribute cache, and — since [`Substrate::partition`] exists — possibly
    /// through a client that cannot reach the server at all. The server's own tree is the only view
    /// that is true regardless of what has been done to the clients.
    pub shards: Vec<(String, PathBuf)>,
    /// Per client, the shard paths that client reaches its shards through. One entry per replica
    /// under NFS; under a local directory every client shares the one set, because there is no
    /// client to speak of.
    clients: Vec<Vec<(String, PathBuf)>>,
    /// The private NFS client each replica reaches its shards through. Empty on a local directory.
    nets: Vec<NetClient>,
    /// What the NFS substrate has to undo, once.
    nfs_teardown: Option<Undo>,
    /// Kept so the temporary root outlives the run.
    _root: tempfile::TempDir,
}

/// The /30 client `c` is wired on: `.1` this host, `.2` the namespace.
///
/// The second octet is derived from the pid so two simulators on one machine do not wire themselves
/// into each other's subnets — the same reason the export takes a distinct `fsid`.
fn client_subnet(c: usize) -> (String, String) {
    let octet = 66 + (std::process::id() as usize % 60);
    (format!("10.{octet}.{c}.1"), format!("10.{octet}.{c}.2"))
}

impl Drop for Substrate {
    fn drop(&mut self) {
        // A simulator that leaves NFS mounts behind has damaged the machine it was borrowing. Every
        // step is best-effort and in reverse order: unmount each shard, then withdraw the export. A
        // lazy unmount (`-l`) because a replica that outlived its scenario may still hold a
        // descriptor, and a mount that will not come down is worse than one that comes down late.
        let Some(undo) = self.nfs_teardown.take() else {
            return;
        };
        if let Ok(mut pending) = PENDING.lock() {
            pending.retain(|u| u.export != undo.export);
        }
        teardown(&undo);
    }
}

impl Substrate {
    /// Prepare `shards` named `s1..sN`, reachable by `clients` independent NFS clients.
    ///
    /// `clients` is the replica count: under NFS each replica mounts the export separately, from its
    /// own loopback address, so one replica's storage can be taken away without touching its peers'.
    /// Point at a shared filesystem somebody else provisioned.
    ///
    /// Validated rather than assumed: every path must exist, be a directory, and be **writable** —
    /// an EFS access point with the wrong uid produces a mount the replicas cannot use, and finding
    /// that out from a scenario failing three steps later wastes a run. The probe file is written and
    /// removed here for the same reason `/readyz` writes one: a directory that stats fine and rejects
    /// a create is the failure mode worth catching up front.
    pub fn attach(shards: Vec<(String, PathBuf)>) -> Result<Self, String> {
        if shards.is_empty() {
            return Err("an attached substrate needs at least one --shard name=/path".into());
        }
        for (name, path) in &shards {
            if !path.is_dir() {
                return Err(format!(
                    "shard {name} at {} is not a directory that exists",
                    path.display()
                ));
            }
            let probe = path.join(".fleet-sim-writable");
            std::fs::write(&probe, b"")
                .map_err(|e| format!("shard {name} at {} is not writable: {e}", path.display()))?;
            let _ = std::fs::remove_file(&probe);
        }
        Ok(Self {
            kind: Kind::Attached,
            // Every replica reaches the same filesystem at the same path — an EFS volume is mounted
            // into each task at the same container path, so there is one set, shared.
            clients: vec![shards.clone()],
            shards,
            nets: Vec::new(),
            nfs_teardown: None,
            _root: tempfile::tempdir().map_err(|e| format!("scratch dir: {e}"))?,
        })
    }

    pub fn prepare(kind: Kind, shards: usize, clients: usize) -> Result<Self, String> {
        let clients = clients.max(1);
        if kind == Kind::Nfs {
            preflight_nfs()?;
            return Self::prepare_nfs(shards, clients);
        }
        Self::prepare_local(shards, clients)
    }

    fn prepare_local(shards: usize, clients: usize) -> Result<Self, String> {
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
            // One directory, however many "clients": a local filesystem has exactly one, which is
            // the whole reason this substrate cannot settle the cross-client claims.
            clients: vec![out.clone(); clients],
            shards: out,
            nets: Vec::new(),
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
    fn prepare_nfs(shards: usize, clients: usize) -> Result<Self, String> {
        let root = tempfile::tempdir().map_err(|e| format!("substrate root: {e}"))?;
        let export = root.path().join("export");
        let mnt = root.path().join("mnt");
        std::fs::create_dir_all(&export).map_err(|e| format!("export dir: {e}"))?;
        std::fs::create_dir_all(&mnt).map_err(|e| format!("mount dir: {e}"))?;

        // The server's own tree, which is what the checker reads.
        let mut served = Vec::new();
        for i in 1..=shards {
            let name = format!("s{i}");
            std::fs::create_dir_all(export.join(&name))
                .map_err(|e| format!("shard {name}: {e}"))?;
            served.push((name.clone(), export.join(&name)));
        }
        // And one mount point per client per shard.
        let mut per_client: Vec<Vec<(String, PathBuf)>> = Vec::new();
        for c in 0..clients {
            let mut mine = Vec::new();
            for (name, _) in &served {
                let at = mnt.join(format!("c{c}")).join(name);
                std::fs::create_dir_all(&at).map_err(|e| format!("mnt {name}: {e}"))?;
                mine.push((name.clone(), at));
            }
            per_client.push(mine);
        }
        let all_mounts: Vec<PathBuf> = per_client
            .iter()
            .flat_map(|m| m.iter().map(|(_, p)| p.clone()))
            .collect();

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
            // Offered to the private range the veth pairs live on (see [`NetClient`]), not to
            // loopback: each replica reaches the server from its own namespace. Nothing outside this
            // host is routed there.
            &format!("{EXPORT_NET}:{}", export.display()),
        ])?;

        // Registered before anything is created: from here on a signal tears down whatever exists,
        // and the list is grown as each namespace appears so a failure halfway leaves nothing.
        let pid = std::process::id();
        let mut nets: Vec<NetClient> = Vec::new();
        if let Ok(mut pending) = PENDING.lock() {
            pending.push(Undo {
                export: export.clone(),
                mounts: all_mounts.clone(),
                nets: Vec::new(),
            });
        }

        for (c, mine) in per_client.iter().enumerate() {
            let net = NetClient {
                ns: format!("fsim{pid}c{c}"),
                host_if: format!("fh{pid}c{c}"),
                client_if: format!("fc{pid}c{c}"),
                host_ip: client_subnet(c).0,
            };
            let (host_ip, client_ip) = client_subnet(c);

            // Registered for teardown **before** it is created, not after. Deleting a namespace or
            // a link that was never made is a no-op, so registering early costs nothing — while
            // registering afterwards leaves a window in which a failure between the two lines
            // strands a namespace nobody will ever delete.
            if let Ok(mut pending) = PENDING.lock()
                && let Some(undo) = pending.last_mut()
            {
                undo.nets.push((net.ns.clone(), net.host_if.clone()));
            }

            sudo(&["ip", "netns", "add", &net.ns])?;
            sudo(&[
                "ip",
                "link",
                "add",
                &net.host_if,
                "type",
                "veth",
                "peer",
                "name",
                &net.client_if,
            ])?;
            sudo(&["ip", "link", "set", &net.client_if, "netns", &net.ns])?;
            sudo(&[
                "ip",
                "addr",
                "add",
                &format!("{host_ip}/30"),
                "dev",
                &net.host_if,
            ])?;
            sudo(&["ip", "link", "set", &net.host_if, "up"])?;
            for args in [
                vec![
                    "ip",
                    "addr",
                    "add",
                    &format!("{client_ip}/30"),
                    "dev",
                    &net.client_if,
                ],
                vec!["ip", "link", "set", &net.client_if, "up"],
                vec!["ip", "link", "set", "lo", "up"],
            ] {
                let mut full = vec!["ip", "netns", "exec", &net.ns];
                full.extend(args.iter().copied());
                sudo(&full)?;
            }

            for (name, path) in mine {
                // The mount is made **inside** the network and UTS namespaces and **outside** any
                // new mount namespace. The UTS hostname is what gives this client its own NFSv4
                // identity (and so its own lease); the network namespace is what the RPC travels
                // through; and leaving the mount namespace alone is what makes the result visible
                // to an ordinary replica process on this host. `nsenter`/`unshare` rather than
                // `ip netns exec`, which unshares mounts and would take the mount with it when it
                // exited.
                let script = format!(
                    "hostname {ns} && mount -t nfs4 -o \
                     nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport \
                     {host_ip}:{}/{name} {}",
                    export.display(),
                    path.display(),
                    ns = net.ns,
                );
                sudo(&[
                    "nsenter",
                    &format!("--net=/var/run/netns/{}", net.ns),
                    "unshare",
                    "--uts",
                    "sh",
                    "-c",
                    &script,
                ])?;
            }
            nets.push(net);
        }

        Ok(Self {
            kind: Kind::Nfs,
            nfs_teardown: Some(Undo {
                export,
                mounts: all_mounts,
                nets: nets
                    .iter()
                    .map(|n| (n.ns.clone(), n.host_if.clone()))
                    .collect(),
            }),
            nets,
            clients: per_client,
            shards: served,
            _root: root,
        })
    }

    /// The `--shard` arguments replica `client` is launched with.
    ///
    /// Each replica gets **its own mount** of the same export, so "this replica's storage stopped
    /// answering" is a thing that can happen to one of them.
    pub fn shard_args_for(&self, client: usize) -> Vec<(&str, &Path)> {
        let mine = &self.clients[client.min(self.clients.len() - 1)];
        mine.iter()
            .map(|(n, p)| (n.as_str(), p.as_path()))
            .collect()
    }

    /// Take `client`'s storage away, the way a lost mount target does: packets to its NFS server
    /// address are dropped, so a `hard` mount blocks rather than erroring, and its lease stops being
    /// renewed. Its peers are untouched.
    ///
    /// Idempotent — partitioning an already-partitioned client is a no-op, because the rule is
    /// inserted only if it isn't there.
    pub fn partition(&self, client: usize) -> Result<(), String> {
        self.link(client, "down")
    }

    /// Give `client` its storage back. Idempotent: bringing an already-up link up is a no-op, so a
    /// client is healed however many times it was partitioned — including by `Drop`, which heals
    /// before it unmounts.
    pub fn heal(&self, client: usize) {
        let _ = self.link(client, "up");
    }

    /// The link's state is the partition. Taking this replica's end of the veth down stops its NFS
    /// traffic *and* its lease renewals without touching its peers, which is what a lost mount
    /// target does; a firewall rule on a shared address could not, because a shared address is a
    /// shared client.
    fn link(&self, client: usize, state: &str) -> Result<(), String> {
        if self.kind == Kind::Attached {
            return Err(
                "an attached substrate's network belongs to whoever provisioned it — partition it \
                 through the fault command instead"
                    .into(),
            );
        }
        if self.kind != Kind::Nfs {
            return Err("a local directory cannot be partitioned from itself".into());
        }
        let Some(net) = self.nets.get(client) else {
            return Err(format!("no such client: {client}"));
        };
        sudo(&[
            "ip",
            "netns",
            "exec",
            &net.ns,
            "ip",
            "link",
            "set",
            &net.client_if,
            state,
        ])
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
