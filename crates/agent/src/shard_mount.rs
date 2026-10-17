//! What the epoch fence needs from the NFS mount a shard lives on, checked once at service-mode
//! startup ([`check_shards`]).
//!
//! The segmented session layout (`session_store`, "Layout::Segmented") fences owners with two
//! filesystem checks. **Taking a session over** (`ensure_target`) reads the predecessor segment and
//! `O_EXCL`-creates the next epoch: the create is always a round trip to the server, and under
//! close-to-open consistency the open that read the predecessor revalidated its attributes and data.
//! **Staying the owner** (`guard_not_superseded`) looks up the next epoch's name before each
//! committed batch — and on NFS a *negative* lookup is cached (a negative dentry) and trusted until
//! the directory's cached attributes expire, which takes up to `acdirmax`.
//!
//! A successor can only create that next epoch once the owner's session lock is free: released (the
//! owner stopped writing), or — the case the fence exists for, an owner cut off from storage — its
//! NFSv4 lease expired on the server, at least one lease after the owner's last contact. Every
//! attribute the owner has cached was fetched no later than that contact, so when it comes back and
//! checks, the cache that could hide the successor's segment is already more than one lease old: it
//! has expired if the directory attribute timeout is shorter than the lease. Hence the rule:
//!
//! - **`nocto`** is refused outright: without close-to-open a successor's open of the predecessor
//!   may be served from cache, and the successor seals it short of lines its owner acknowledged.
//! - **`lookupcache=none` / `lookupcache=pos[itive]`**: negative lookups are always revalidated, so
//!   no attribute timeout can hide a successor and `acdirmax` is not limited.
//! - Otherwise (`lookupcache=all`, the default): the directory attribute timeout — the larger of
//!   `acdirmin` and `acdirmax`, both of which `actimeo` sets — must be below [`ASSUMED_LEASE`] less
//!   [`LEASE_MARGIN`]: under 80 s. The defaults (30 s / 60 s, which the kernel leaves out of
//!   `mountinfo`) pass, and so does every EFS mount made with them (ECS/Fargate's always is).
//!
//! The lease is the server's and no client interface exposes it, so the rule assumes the 90 s that
//! both EFS and Linux `nfsd` use by default. A server configured with a shorter lease needs a
//! correspondingly smaller `acdirmax` — this check cannot see that. A shard that is not on NFS has
//! no client attribute cache and passes.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::service::Shards;

/// The NFSv4 lease the rule assumes: EFS's, and Linux `nfsd`'s default (`nfsv4leasetime`).
pub const ASSUMED_LEASE: Duration = Duration::from_secs(90);
/// Slack under the lease for the client's and server's clocks and for when the client stamps a
/// cached attribute.
pub const LEASE_MARGIN: Duration = Duration::from_secs(10);
/// The kernel's defaults, used when `mountinfo` leaves an option out (it prints only non-defaults).
const DEFAULT_ACDIRMIN: u64 = 30;
const DEFAULT_ACDIRMAX: u64 = 60;

/// One `/proc/self/mountinfo` line, as much of it as the rule reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mount {
    pub mount_point: PathBuf,
    pub fs_type: String,
    /// The per-mount options and the superblock options, in that order (NFS puts its cache
    /// settings in the latter).
    pub options: Vec<String>,
}

impl Mount {
    fn option(&self, key: &str) -> Option<&str> {
        // The last occurrence wins, as on a mount command line.
        self.options
            .iter()
            .rev()
            .find_map(|o| o.strip_prefix(key).and_then(|rest| rest.strip_prefix('=')))
    }

    fn flag(&self, name: &str) -> bool {
        self.options.iter().any(|o| o == name)
    }

    fn seconds(&self, key: &str) -> Result<Option<u64>, String> {
        self.option(key)
            .map(|v| {
                v.parse()
                    .map_err(|_| format!("{key}={v} is not a number of seconds"))
            })
            .transpose()
    }
}

/// Parse `/proc/self/mountinfo`. A line that does not have the documented shape is skipped.
pub fn parse_mountinfo(text: &str) -> Vec<Mount> {
    text.lines().filter_map(parse_line).collect()
}

/// `36 35 98:0 /mnt1 /mnt/parent rw,noatime master:1 - ext3 /dev/root rw,errors=continue`: the
/// mount point is field 5, the per-mount options field 6, and after the ` - ` separator come the
/// filesystem type, the source and the superblock options.
fn parse_line(line: &str) -> Option<Mount> {
    let (head, tail) = line.split_once(" - ")?;
    let head: Vec<&str> = head.split(' ').collect();
    let mount_point = unescape(head.get(4)?);
    let mount_opts = head.get(5)?;
    let mut tail = tail.split(' ');
    let fs_type = tail.next()?.to_string();
    let _source = tail.next()?;
    let super_opts = tail.next().unwrap_or("");
    let options = mount_opts
        .split(',')
        .chain(super_opts.split(','))
        .filter(|o| !o.is_empty())
        .map(str::to_string)
        .collect();
    Some(Mount {
        mount_point: PathBuf::from(mount_point),
        fs_type,
        options,
    })
}

/// `mountinfo` escapes space, tab, newline and backslash as `\ooo` octal.
fn unescape(field: &str) -> String {
    let bytes = field.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && let Some(oct) = bytes.get(i + 1..i + 4)
            && oct.iter().all(|b| (b'0'..=b'7').contains(b))
        {
            out.push((oct[0] - b'0') * 64 + (oct[1] - b'0') * 8 + (oct[2] - b'0'));
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The mount `path` lives on: the last-mounted of those with the longest mount point above it
/// (a later mount on the same point hides an earlier one).
pub fn mount_for<'a>(path: &Path, mounts: &'a [Mount]) -> Option<&'a Mount> {
    mounts
        .iter()
        .enumerate()
        .filter(|(_, m)| path.starts_with(&m.mount_point))
        .max_by_key(|(i, m)| (m.mount_point.components().count(), *i))
        .map(|(_, m)| m)
}

/// Whether the epoch fence holds on `mount` (see the module doc). `Err` says which option breaks
/// it and why.
pub fn check_mount(mount: &Mount) -> Result<(), String> {
    if !matches!(mount.fs_type.as_str(), "nfs" | "nfs4") {
        return Ok(());
    }
    if mount.flag("nocto") {
        return Err(
            "it is mounted with nocto: without close-to-open consistency a new owner can read a \
             predecessor segment from cache and seal it short of lines its owner already \
             acknowledged. Remount without nocto"
                .into(),
        );
    }
    if matches!(
        mount.option("lookupcache"),
        Some("none" | "pos" | "positive")
    ) {
        return Ok(());
    }
    let actimeo = mount.seconds("actimeo")?;
    let acdirmin = mount
        .seconds("acdirmin")?
        .or(actimeo)
        .unwrap_or(DEFAULT_ACDIRMIN);
    let acdirmax = mount
        .seconds("acdirmax")?
        .or(actimeo)
        .unwrap_or(DEFAULT_ACDIRMAX);
    let timeout = acdirmin.max(acdirmax);
    let limit = (ASSUMED_LEASE - LEASE_MARGIN).as_secs();
    if timeout >= limit {
        return Err(format!(
            "its directory attribute cache lasts up to {timeout} s (acdirmin={acdirmin}, \
             acdirmax={acdirmax}), and the session epoch fence needs under {limit} s — the {} s NFSv4 \
             lease less a {} s margin — or a cached negative lookup can hide a new owner's segment \
             and this replica goes on acknowledging turns nobody will replay. Remount with \
             acdirmax (and actimeo) below {limit}, or with lookupcache=positive",
            ASSUMED_LEASE.as_secs(),
            LEASE_MARGIN.as_secs(),
        ));
    }
    Ok(())
}

/// Check every shard's mount against the fence's rule, given `mountinfo`'s text. A shard path is
/// resolved through symlinks first when it exists, so the mount found is the one it really lives on.
pub fn check_shards_with(shards: &Shards, mountinfo: &str) -> Result<(), String> {
    let mounts = parse_mountinfo(mountinfo);
    for (name, path) in shards.iter() {
        let real = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let Some(mount) = mount_for(&real, &mounts) else {
            continue;
        };
        check_mount(mount).map_err(|why| {
            format!(
                "--shard {name}={}: refusing to serve from the {} mount at {}: {why}",
                path.display(),
                mount.fs_type,
                mount.mount_point.display()
            )
        })?;
    }
    Ok(())
}

/// [`check_shards_with`] on this process's own mounts. Fails closed when they cannot be read: the
/// check is the only thing standing between a misconfigured mount and lost acknowledged writes.
#[cfg(target_os = "linux")]
pub fn check_shards(shards: &Shards) -> Result<(), String> {
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo").map_err(|e| {
        format!("--service: cannot read /proc/self/mountinfo to check the shard mounts: {e}")
    })?;
    check_shards_with(shards, &mountinfo)
}

/// Not Linux: no `mountinfo`, and service mode's NFS deployments are all Linux.
#[cfg(not(target_os = "linux"))]
pub fn check_shards(_shards: &Shards) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An EFS mount as ECS/Fargate makes it: kernel defaults for every cache timeout, so none of
    /// them is printed.
    const EFS: &str = "1424 1390 0:131 / /mnt/efs rw,relatime shared:640 - nfs4 \
                       127.0.0.1:/ rw,vers=4.1,rsize=1048576,wsize=1048576,namlen=255,hard,\
                       noresvport,proto=tcp,port=20049,timeo=600,retrans=2,sec=sys,\
                       clientaddr=127.0.0.1,local_lock=none,addr=127.0.0.1";
    const ROOT: &str =
        "22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw,errors=remount-ro";

    fn nfs(super_opts: &str) -> String {
        format!(
            "{ROOT}\n1500 22 0:140 / /mnt/nfs rw,relatime shared:700 - nfs4 server:/export \
             rw,vers=4.2,hard,proto=tcp,{super_opts}"
        )
    }

    fn shards(path: &str) -> Shards {
        Shards::parse(&[format!("s1={path}")]).unwrap()
    }

    fn check(mountinfo: &str, path: &str) -> Result<(), String> {
        check_shards_with(&shards(path), mountinfo)
    }

    #[test]
    fn the_efs_defaults_pass() {
        let text = format!("{ROOT}\n{EFS}");
        assert_eq!(check(&text, "/mnt/efs/s1"), Ok(()));
        let m = &parse_mountinfo(&text)[1];
        assert_eq!(m.mount_point, Path::new("/mnt/efs"));
        assert_eq!(m.fs_type, "nfs4");
    }

    #[test]
    fn a_shard_on_a_local_disk_passes() {
        assert_eq!(check(ROOT, "/srv/shards/s1"), Ok(()));
    }

    #[test]
    fn nocto_is_refused() {
        let err = check(&nfs("nocto"), "/mnt/nfs/s1").unwrap_err();
        assert!(err.contains("--shard s1=/mnt/nfs/s1"), "{err}");
        assert!(err.contains("nocto"), "{err}");
    }

    #[test]
    fn a_directory_cache_as_long_as_the_lease_less_the_margin_is_refused() {
        // `actimeo=120` is printed by the kernel as all four timeouts.
        let err = check(
            &nfs("acregmin=120,acregmax=120,acdirmin=120,acdirmax=120"),
            "/mnt/nfs/s1",
        )
        .unwrap_err();
        assert!(err.contains("acdirmax=120"), "{err}");
        assert!(err.contains("under 80 s"), "{err}");
        assert!(
            check(&nfs("acdirmax=80"), "/mnt/nfs/s1").is_err(),
            "the bound is exclusive"
        );
        assert_eq!(check(&nfs("acdirmax=79"), "/mnt/nfs/s1"), Ok(()));
        // `actimeo` itself, should a kernel print it.
        assert!(check(&nfs("actimeo=300"), "/mnt/nfs/s1").is_err());
        // An acdirmin above the limit is the timeout even when acdirmax is not.
        assert!(check(&nfs("acdirmin=90,acdirmax=60"), "/mnt/nfs/s1").is_err());
        // `noac`: every timeout 0.
        assert_eq!(
            check(
                &nfs("sync,acregmin=0,acregmax=0,acdirmin=0,acdirmax=0"),
                "/mnt/nfs/s1"
            ),
            Ok(())
        );
    }

    #[test]
    fn without_negative_lookup_caching_acdirmax_is_not_limited() {
        for lc in [
            "lookupcache=none",
            "lookupcache=pos",
            "lookupcache=positive",
        ] {
            assert_eq!(
                check(&nfs(&format!("acdirmax=600,{lc}")), "/mnt/nfs/s1"),
                Ok(()),
                "{lc}"
            );
        }
        // `all` is the default: still limited.
        assert!(check(&nfs("acdirmax=600,lookupcache=all"), "/mnt/nfs/s1").is_err());
        // nocto is refused whatever lookupcache says.
        assert!(check(&nfs("nocto,lookupcache=none"), "/mnt/nfs/s1").is_err());
    }

    #[test]
    fn the_deepest_and_latest_mount_over_the_shard_decides() {
        // A local bind over the NFS mount's subdirectory hides it for that subtree only.
        let text = format!(
            "{}\n1600 1500 259:2 /x /mnt/nfs/local rw - ext4 /dev/nvme0n1p2 rw",
            nfs("acdirmax=600")
        );
        assert_eq!(check(&text, "/mnt/nfs/local/s1"), Ok(()));
        assert!(check(&text, "/mnt/nfs/s1").is_err());
        // A sibling whose name merely starts the same is not under it.
        assert_eq!(check(&text, "/mnt/nfsother/s1"), Ok(()));
        // Mounted again on the same point: the later one is what the path reaches.
        let again = format!("{text}\n1700 22 259:2 / /mnt/nfs rw - ext4 /dev/nvme0n1p2 rw");
        assert_eq!(check(&again, "/mnt/nfs/s1"), Ok(()));
    }

    #[test]
    fn escaped_mount_points_and_malformed_lines() {
        let text = "1500 22 0:140 / /mnt/my\\040nfs rw - nfs4 server:/ rw,nocto\nnot a mount line";
        let mounts = parse_mountinfo(text);
        assert_eq!(mounts.len(), 1);
        assert_eq!(mounts[0].mount_point, Path::new("/mnt/my nfs"));
        assert!(check(text, "/mnt/my nfs/s1").is_err());
        assert!(check(&nfs("acdirmax=soon"), "/mnt/nfs/s1").is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn this_hosts_own_mounts_are_read() {
        let dir = tempfile::tempdir().unwrap();
        let shards = shards(&dir.path().to_string_lossy());
        // Whatever the temp dir is on here, reading and parsing the real file works.
        let text = std::fs::read_to_string("/proc/self/mountinfo").unwrap();
        assert!(!parse_mountinfo(&text).is_empty());
        assert_eq!(
            check_shards(&shards),
            Ok(()),
            "a temp dir is not on NFS here"
        );
    }
}
