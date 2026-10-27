//! Ground truth: the real filesystem, read independently with `std::fs`.
//!
//! Nothing here goes through `ShellFs`, `LocalFs` or any exec path. `snapshot` reads the whole tree
//! (temporarily widening permissions on entries the test itself made unreadable, then restoring them),
//! and `plan_write`/`plan_mkdir` predict from the tree alone whether an operation must succeed, must
//! fail, and with which spec error kind.
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use beyond_agent::tools::fs::FsError;

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    File {
        mode: u32,
        mtime: SystemTime,
        bytes: Vec<u8>,
    },
    Dir {
        mode: u32,
        kids: BTreeSet<OsString>,
    },
    Link(PathBuf),
    Other {
        mode: u32,
    },
}

pub type Snap = BTreeMap<PathBuf, Node>;

/// The spec's error vocabulary (FILES-OVER-EXEC section 3.1), as far as `ShellFs` can tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    NotFound,
    IsDirectory,
    NotADirectory,
    PermissionDenied,
    Io,
    TooLarge,
    Transport,
    Timeout,
    InvalidQuery,
}

/// Classify a `ShellFs` error the way FILES-OVER-EXEC section 3.3 does, from its text. Every path the
/// operation named is removed first: a file called `is a directory` must not change the answer.
pub fn classify(e: &FsError, paths: &[&Path]) -> Kind {
    let msg = match e {
        FsError::InvalidQuery(_) => return Kind::InvalidQuery,
        FsError::Backend(m) => m.clone(),
    };
    let mut m = msg;
    let mut ps: Vec<String> = paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    ps.sort_by_key(|p| std::cmp::Reverse(p.len()));
    for p in ps {
        if !p.is_empty() {
            m = m.replace(&p, "");
        }
    }
    let m = m.to_lowercase();
    if m.contains("exceeded the") && m.contains("-byte cap") {
        Kind::TooLarge
    } else if m.contains("timed out") {
        Kind::Timeout
    } else if m.contains("unreachable") || m.contains("injected") {
        Kind::Transport
    } else if m.contains("no such file or directory")
        || m.contains("directory nonexistent")
        || m.contains("nonexistent directory")
    {
        Kind::NotFound
    } else if m.contains("is a directory") {
        Kind::IsDirectory
    } else if m.contains("not a directory") {
        Kind::NotADirectory
    } else if m.contains("permission denied") {
        Kind::PermissionDenied
    } else {
        Kind::Io
    }
}

pub fn my_uid(probe: &Path) -> u32 {
    std::fs::metadata(probe).map(|m| m.uid()).unwrap_or(0)
}

pub fn is_root() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("Uid:").map(str::to_string))
        })
        .and_then(|l| l.split_whitespace().nth(1).map(|u| u == "0"))
        .unwrap_or(false)
}

/// Whether the calling user may do `bits` (4 read, 2 write, 1 execute) to `p`, by the owner/other mode
/// bits of an existing path. Everything the tests create is owned by the caller; the one exception
/// they touch is `/dev/null`, which is world-accessible.
pub fn can(p: &Path, bits: u32) -> bool {
    if is_root() {
        return true;
    }
    let Ok(m) = std::fs::metadata(p) else {
        return false;
    };
    let mode = m.mode();
    let me = unsafe_free_uid();
    let class = if m.uid() == me { mode >> 6 } else { mode };
    class & bits == bits
}

fn unsafe_free_uid() -> u32 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("Uid:").map(str::to_string))
        })
        .and_then(|l| l.split_whitespace().nth(1).and_then(|u| u.parse().ok()))
        .unwrap_or(0)
}

/// Read the whole tree. Entries the test made unreadable are widened for the read and restored.
pub fn snapshot(root: &Path) -> Snap {
    let mut out = Snap::new();
    walk(root, &mut out);
    out
}

fn walk(p: &Path, out: &mut Snap) {
    let Ok(m) = std::fs::symlink_metadata(p) else {
        return;
    };
    let mode = m.mode() & 0o7777;
    let ft = m.file_type();
    if ft.is_symlink() {
        out.insert(
            p.to_path_buf(),
            Node::Link(std::fs::read_link(p).unwrap_or_default()),
        );
    } else if ft.is_dir() {
        let widened = mode & 0o700 != 0o700;
        if widened {
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode | 0o700));
        }
        let mut kids = BTreeSet::new();
        let mut paths = Vec::new();
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                kids.insert(e.file_name());
                paths.push(e.path());
            }
        }
        for c in paths {
            walk(&c, out);
        }
        if widened {
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode));
        }
        out.insert(p.to_path_buf(), Node::Dir { mode, kids });
    } else if ft.is_file() {
        let widened = mode & 0o600 != 0o600;
        if widened {
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode | 0o600));
        }
        let bytes = std::fs::read(p).unwrap_or_default();
        if widened {
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode));
        }
        out.insert(
            p.to_path_buf(),
            Node::File {
                mode,
                mtime: m.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                bytes,
            },
        );
    } else {
        out.insert(p.to_path_buf(), Node::Other { mode });
    }
}

/// A name ShellFs's write temp would have: `<name>.tmp.<hex>`.
pub fn looks_like_temp(name: &std::ffi::OsStr) -> bool {
    let b = name.to_os_string().into_vec();
    let s = String::from_utf8_lossy(&b);
    match s.rsplit_once(".tmp.") {
        Some((_, hex)) => hex.len() >= 8 && hex.bytes().all(|c| c.is_ascii_hexdigit()),
        None => false,
    }
}

pub fn temps(s: &Snap) -> Vec<PathBuf> {
    s.keys()
        .filter(|p| p.file_name().is_some_and(looks_like_temp))
        .cloned()
        .collect()
}

/// `canonicalize`, except under `/proc/PID/root`, whose magic link would send it to the host's own
/// view of the path instead of the namespace's.
pub fn canon(p: &Path) -> std::io::Result<PathBuf> {
    if p.starts_with("/proc") {
        Ok(p.to_path_buf())
    } else {
        std::fs::canonicalize(p)
    }
}

/// Follow the symlinks at the final component, as the write scripts do (hop by hop, at most 40). `None` for a loop.
pub fn resolve_leaf(p: &Path) -> Option<PathBuf> {
    // Directories on the way are followed by the kernel; the final component by the script.
    let canon_parent = |q: &Path| -> Option<PathBuf> {
        let name = q.file_name()?;
        let parent = q.parent()?;
        Some(
            canon(parent)
                .unwrap_or_else(|_| parent.to_path_buf())
                .join(name),
        )
    };
    let mut cur = canon_parent(p)?;
    for _ in 0..40 {
        match std::fs::symlink_metadata(&cur) {
            Ok(m) if m.file_type().is_symlink() => {
                let t = std::fs::read_link(&cur).ok()?;
                let next = if t.is_absolute() {
                    t
                } else {
                    cur.parent()?.join(t)
                };
                cur = canon_parent(&next)?;
            }
            _ => return Some(cur),
        }
    }
    None
}

#[derive(Clone, Debug)]
pub enum Plan {
    /// The spec requires this to succeed.
    Succeed,
    /// The spec requires this to fail, with one of these kinds.
    Fail(Vec<Kind>),
    /// The spec does not decide (a name too close to NAME_MAX for the temp's suffix).
    Either,
}

/// Longest temp suffix `temp_suffix()` makes: 16 hex digits of salt plus a hex counter.
const SUFFIX_MAX: usize = 22;
const SUFFIX_MIN: usize = 5;

pub fn plan_write(path: &Path) -> Plan {
    let Some(t) = resolve_leaf(path) else {
        return Plan::Either;
    };
    if let Ok(m) = std::fs::metadata(&t)
        && m.is_dir()
    {
        // D1: a write onto a directory must fail and change nothing.
        return Plan::Fail(vec![Kind::IsDirectory, Kind::Io]);
    }
    let Some(parent) = t.parent() else {
        return Plan::Fail(vec![Kind::Io]);
    };
    match std::fs::metadata(parent) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // A missing parent, or one whose own parent cannot be searched (the shell reads both
            // as "No such file or directory").
            return Plan::Fail(vec![Kind::NotFound, Kind::PermissionDenied]);
        }
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Plan::Fail(vec![Kind::PermissionDenied, Kind::NotFound]);
        }
        Err(_) => return Plan::Fail(vec![Kind::NotADirectory, Kind::NotFound]),
        // dash words a redirect below a file as "Directory nonexistent", GNU tools as ENOTDIR.
        Ok(m) if !m.is_dir() => return Plan::Fail(vec![Kind::NotADirectory, Kind::NotFound]),
        Ok(_) => {}
    }
    // A temp name past NAME_MAX fails at lookup, before the permission to create it is asked.
    let name_len = t.file_name().map_or(0, |n| n.len());
    if name_len + SUFFIX_MAX + 5 > 255 {
        return if name_len + SUFFIX_MIN + 5 > 255 {
            Plan::Fail(vec![Kind::Io, Kind::PermissionDenied])
        } else {
            Plan::Either
        };
    }
    if !can(parent, 0o3) {
        return Plan::Fail(vec![Kind::PermissionDenied]);
    }
    let exists = std::fs::symlink_metadata(&t).is_ok();
    if exists && std::fs::metadata(&t).is_ok() && !can(&t, 0o2) {
        return Plan::Fail(vec![Kind::PermissionDenied]);
    }
    Plan::Succeed
}

pub fn plan_mkdir(path: &Path) -> Plan {
    if let Ok(m) = std::fs::metadata(path) {
        return if m.is_dir() {
            Plan::Succeed
        } else {
            Plan::Fail(vec![Kind::Io, Kind::NotADirectory])
        };
    }
    if std::fs::symlink_metadata(path).is_ok() {
        return Plan::Either; // a dangling link
    }
    let mut cur = path.parent();
    while let Some(c) = cur {
        match std::fs::metadata(c) {
            Ok(m) if m.is_dir() => {
                return if can(c, 0o3) {
                    Plan::Succeed
                } else {
                    Plan::Fail(vec![Kind::PermissionDenied])
                };
            }
            Ok(_) => return Plan::Fail(vec![Kind::NotADirectory, Kind::Io]),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if std::fs::symlink_metadata(c).is_ok() {
                    // A dangling link on the way: `mkdir -p` cannot make it a directory.
                    return Plan::Fail(vec![Kind::Io, Kind::NotFound]);
                }
                cur = c.parent()
            }
            Err(_) => return Plan::Either,
        }
    }
    Plan::Either
}

pub fn secs(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
