//! The agent as a deployment: a grant's [`Bundle`] is an immutable plain tar of the agent folder
//! (`AGENTS.md`, `.mcp.json`, `.agents/skills`, `.agents/agents`, ...), named by the SHA-256 of its
//! bytes the way an image is named by its digest.
//!
//! [`Cache::ensure`] fetches it through the replica's egress client, checks the hash **before** a byte is
//! unpacked, and unpacks it into a cache directory named by the hash, built beside it and renamed into
//! place, so a directory that exists is always a complete, verified copy. Every session on the replica
//! that names the same hash shares that one copy and fetches nothing when it is there. A fetch, hash or
//! unpack failure is the session's open failing: nothing falls back to another definition.
//!
//! The format is plain `ustar` (uncompressed: the transport can compress, and a size cap on bytes that
//! are the bytes on disk is a real cap). Unpacking writes regular files and directories only. A path
//! that is absolute or climbs with `..`, a symlink, a hardlink, a device, a pax or GNU extension
//! header, or more than [`MAX_BYTES`] of content refuses the whole bundle; archived modes, owners and
//! times are ignored. [`BundleFs`] then serves the copy read-only at the grant's `workspace_root`, so
//! the definition's paths mean what they would in a sandbox that holds the same files.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::grant::Bundle;
use crate::tools::fs::local::LocalFs;
use crate::tools::fs::{
    DirEntry, FsBackend, FsError, GlobOutcome, GlobQuery, Meta, PathWorld, SearchOutcome,
    SearchQuery,
};

/// The most a bundle may weigh, as fetched and as unpacked.
pub const MAX_BYTES: usize = 64 * 1024 * 1024;

const BLOCK: usize = 512;

/// The default cache budget: 1 GiB. A bundle is an agent folder (text, skills, config), usually
/// kilobytes and at most [`MAX_BYTES`], so this holds thousands of typical versions or sixteen of the
/// largest, while staying a small share of a replica's disk.
pub const DEFAULT_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;

const LOCK: &str = ".lock";

static BUDGET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Set the process cache budget (`--bundle-cache-mb`) before the first session opens.
pub fn set_budget(bytes: u64) {
    let _ = BUDGET.set(bytes);
}

/// How many bundles this process fetches at once. Each body is buffered whole (up to
/// [`MAX_BYTES`]), so this bounds the peak at about 256 MiB however many tenants cold-start at once.
const CONCURRENT_FETCHES: usize = 4;

/// The replica's bundle cache: a directory of its own under `agent-bundles` in its temp dir
/// (`TMPDIR`), within the budget. A cache only: a missing directory is fetched again.
pub fn cache() -> &'static Arc<Cache> {
    static CACHE: std::sync::LazyLock<Arc<Cache>> = std::sync::LazyLock::new(|| {
        Cache::new(
            &std::env::temp_dir().join("agent-bundles"),
            BUDGET.get().copied().unwrap_or(DEFAULT_BUDGET_BYTES),
        )
    });
    &CACHE
}

/// Unpacked bundles on disk, one directory per hash, kept within a byte budget by evicting the least
/// recently released directory nobody holds.
///
/// A session holds a [`Pin`] on its bundle for its whole life (it keeps reading skills from the
/// directory), and a pinned directory is never evicted. So the cache may exceed the budget, but only by
/// the bytes of pinned bundles, and comes back within it as they are released. A bundle's size is the
/// disk it occupies (allocated blocks, directories included), measured once as it unpacks: a bundle of a
/// few small files holds tens of kilobytes of blocks, and counting only content would let thousands of
/// them through a small budget.
///
/// Each cache owns a fresh subdirectory of the shared base, holding a lock file that stays locked for
/// the cache's life. Creating a cache removes the subdirectories of caches whose lock it can take: their
/// process is gone (the kernel releases a crashed process's lock), and no other process's directory is
/// ever touched. A new process starts with an empty cache, and fetching again is cheap and verified.
pub struct Cache {
    /// This cache's own directory.
    root: PathBuf,
    budget: u64,
    fetches: tokio::sync::Semaphore,
    state: std::sync::Mutex<State>,
    /// Holds the lock that tells other processes this directory is in use.
    _lock: Option<std::fs::File>,
}

#[derive(Default)]
struct State {
    entries: HashMap<String, Entry>,
    /// One fetch per hash in flight: the others wait on its lock and find the unpacked copy.
    flights: HashMap<String, Arc<tokio::sync::Mutex<()>>>,
}

struct Entry {
    refs: usize,
    bytes: u64,
    last_release: std::time::SystemTime,
}

/// A hold on one unpacked bundle: its directory stays until this is dropped.
pub struct Pin {
    cache: Arc<Cache>,
    sha256: String,
    dir: PathBuf,
}

impl Pin {
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        let mut state = self.cache.lock();
        if let Some(entry) = state.entries.get_mut(&self.sha256) {
            entry.refs = entry.refs.saturating_sub(1);
            entry.last_release = std::time::SystemTime::now();
        }
        // Releasing is when a cache that grew past its budget on pinned bundles can shrink: with no
        // more installs coming, nothing else would trim it. The directories are renamed away under the
        // lock; deleting them is left to a thread of their own so a drop never waits on the disk.
        let doomed = self.cache.evict(&mut state);
        drop(state);
        if !doomed.is_empty() {
            std::thread::spawn(move || {
                for trash in doomed {
                    let _ = std::fs::remove_dir_all(trash);
                }
            });
        }
    }
}

impl Cache {
    /// A cache in a new subdirectory of `base`, after sweeping the ones their owners have left.
    pub fn new(base: &Path, budget: u64) -> Arc<Self> {
        Self::with_fetch_limit(base, budget, CONCURRENT_FETCHES)
    }

    fn with_fetch_limit(base: &Path, budget: u64, fetches: usize) -> Arc<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let root = base.join(format!(
            "{}-{nanos:x}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        // Lock our own directory before looking at anyone else's, so a sibling starting now sees it
        // as taken.
        let lock = std::fs::create_dir_all(&root).ok().and_then(|()| {
            let file = std::fs::File::create(root.join(LOCK)).ok()?;
            file.try_lock().ok()?;
            Some(file)
        });
        for item in std::fs::read_dir(base).into_iter().flatten().flatten() {
            let theirs = item.path();
            if theirs == root {
                continue;
            }
            // Only a directory whose lock we can take has a dead owner.
            let dead = std::fs::File::open(theirs.join(LOCK)).is_ok_and(|f| f.try_lock().is_ok());
            if dead {
                let _ = std::fs::remove_dir_all(&theirs);
            }
        }
        Arc::new(Self {
            root,
            budget,
            fetches: tokio::sync::Semaphore::new(fetches),
            state: std::sync::Mutex::new(State::default()),
            _lock: lock,
        })
    }

    /// The directory this cache unpacks into.
    pub fn dir(&self) -> &Path {
        &self.root
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Pin the entry for `sha256` if it is cached and still on disk. The `stat` that checks the disk
    /// runs on the blocking pool: on a replica with one async worker, a `stat` stuck behind a slow
    /// filesystem would otherwise stop every other session's open until it returned.
    async fn pin_cached(self: &Arc<Self>, sha256: &str) -> Option<Pin> {
        let (cache, sha256) = (Arc::clone(self), sha256.to_owned());
        // A join error is a panic in the closure or a runtime shutting down: either way, no pin.
        tokio::task::spawn_blocking(move || cache.pin_cached_blocking(&sha256))
            .await
            .ok()
            .flatten()
    }

    fn pin_cached_blocking(self: &Arc<Self>, sha256: &str) -> Option<Pin> {
        let mut state = self.lock();
        let dir = self.root.join(sha256);
        // An entry is removed under this lock before its directory is renamed away, so a missing
        // directory means it was removed from outside: forget the entry.
        if !dir.is_dir() {
            state.entries.remove(sha256);
            return None;
        }
        let entry = state.entries.get_mut(sha256)?;
        entry.refs += 1;
        Some(Pin {
            cache: Arc::clone(self),
            sha256: sha256.to_owned(),
            dir,
        })
    }

    /// The unpacked bundle, pinned, fetching, verifying and unpacking it first if it is not cached.
    pub async fn ensure(
        self: &Arc<Self>,
        bundle: &Bundle,
        egress: Option<&crate::tools::mcp::McpEgress>,
    ) -> Result<Pin, String> {
        if !bundle.is_well_formed() {
            return Err("bundle: the grant's bundle is not a url and a lowercase sha256".into());
        }
        if let Some(pin) = self.pin_cached(&bundle.sha256).await {
            return Ok(pin);
        }
        let egress = egress.ok_or("bundle: this replica has no egress client to fetch it with")?;
        // One fetch per hash in flight in this process: the others wait here and find the unpacked
        // copy. A failed attempt just releases the lock, so the next waiter tries for itself.
        let flight = Arc::clone(
            self.lock()
                .flights
                .entry(bundle.sha256.clone())
                .or_default(),
        );
        let turn = flight.lock().await;
        let result = self.fetch_and_install(bundle, egress).await;
        drop(turn);
        let mut state = self.lock();
        // The map's handle and ours: no waiter has cloned it.
        if Arc::strong_count(&flight) == 2 {
            state.flights.remove(&bundle.sha256);
        }
        drop(state);
        result
    }

    async fn fetch_and_install(
        self: &Arc<Self>,
        bundle: &Bundle,
        egress: &crate::tools::mcp::McpEgress,
    ) -> Result<Pin, String> {
        if let Some(pin) = self.pin_cached(&bundle.sha256).await {
            return Ok(pin);
        }
        let _permit = self.fetches.acquire().await.map_err(|e| e.to_string())?;
        let bytes = egress
            .fetch(&bundle.url, MAX_BYTES)
            .await
            .map_err(|e| format!("bundle: fetching it failed: {e}"))?;
        let got = hex::encode(Sha256::digest(&bytes));
        if got != bundle.sha256 {
            return Err(format!(
                "bundle: its bytes hash to sha256 {got}, not the {} the grant names",
                bundle.sha256
            ));
        }
        let (root, dest) = (self.root.clone(), self.root.join(&bundle.sha256));
        let size = crate::blocking::spawn(crate::blocking::Site::BundleInstall, move || {
            install(&root, &dest, &bytes)
        })
        .await
        .map_err(|e| format!("bundle: unpacking failed: {e}"))??;
        let (pin, doomed) = {
            let mut state = self.lock();
            state.entries.insert(
                bundle.sha256.clone(),
                Entry {
                    refs: 1,
                    bytes: size,
                    last_release: std::time::SystemTime::now(),
                },
            );
            let pin = Pin {
                cache: Arc::clone(self),
                sha256: bundle.sha256.clone(),
                dir: self.root.join(&bundle.sha256),
            };
            (pin, self.evict(&mut state))
        };
        if !doomed.is_empty() {
            let _ = crate::blocking::spawn(crate::blocking::Site::BundleReap, move || {
                for trash in doomed {
                    let _ = std::fs::remove_dir_all(trash);
                }
            })
            .await;
        }
        Ok(pin)
    }

    /// While over budget, drop the unreferenced entry released longest ago. Each directory is renamed
    /// to `.trash-*` here, under the lock, so a request that finds no entry fetches afresh and nothing
    /// can be served from a half-removed directory; the caller deletes the returned paths.
    fn evict(&self, state: &mut State) -> Vec<PathBuf> {
        let mut doomed = Vec::new();
        let mut total: u64 = state.entries.values().map(|e| e.bytes).sum();
        while total > self.budget {
            let Some(oldest) = state
                .entries
                .iter()
                .filter(|(_, e)| e.refs == 0)
                .min_by_key(|(_, e)| e.last_release)
                .map(|(sha, _)| sha.clone())
            else {
                break; // all that is left is pinned
            };
            let Some(entry) = state.entries.remove(&oldest) else {
                break;
            };
            total -= entry.bytes;
            let trash = self
                .root
                .join(format!(".trash-{oldest}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&trash); // a crashed eviction's leftover
            if std::fs::rename(self.root.join(&oldest), &trash).is_ok() {
                doomed.push(trash);
            }
        }
        doomed
    }
}

/// Unpack `bytes` beside `dest` and rename it into place, returning the bytes it occupies on disk.
fn install(root: &Path, dest: &Path, bytes: &[u8]) -> Result<u64, String> {
    let io = |what: &str, e: std::io::Error| format!("bundle: {what}: {e}");
    std::fs::create_dir_all(root).map_err(|e| io("creating the cache", e))?;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = root.join(format!(
        ".unpack-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&tmp); // a crashed process's leftover under a reused pid
    std::fs::create_dir(&tmp).map_err(|e| io("creating a work directory", e))?;
    if let Err(e) = unpack(bytes, &tmp) {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(e);
    }
    let size = disk_bytes(&tmp);
    match std::fs::rename(&tmp, dest) {
        Ok(()) => Ok(size),
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            Err(io("installing it", e))
        }
    }
}

/// The disk `dir` occupies: allocated blocks of every file and directory in it, as `du` counts them.
fn disk_bytes(dir: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt as _;
    let mut total = 0;
    let mut pending = vec![dir.to_path_buf()];
    while let Some(d) = pending.pop() {
        if let Ok(meta) = std::fs::symlink_metadata(&d) {
            total += meta.blocks() * 512;
        }
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            match e.file_type() {
                Ok(t) if t.is_dir() => pending.push(e.path()),
                Ok(_) => total += e.metadata().map_or(0, |m| m.blocks() * 512),
                Err(_) => {}
            }
        }
    }
    total
}

/// Unpack a plain tar into the existing, empty directory `dest`.
pub fn unpack(bytes: &[u8], dest: &Path) -> Result<u64, String> {
    let mut at = 0;
    let mut total = 0usize;
    while let Some(header) = bytes.get(at..at + BLOCK) {
        at += BLOCK;
        if header.iter().all(|b| *b == 0) {
            return Ok(total as u64);
        }
        let path = entry_path(header)?;
        if !checksum_ok(header) {
            return Err(format!("bundle: the header of {path} is corrupt"));
        }
        let size = octal(&header[124..136])
            .ok_or_else(|| format!("bundle: the size of {path} is not a plain number"))?;
        let rel = safe_relative(&path)?;
        match header[156] {
            b'0' | 0 => {
                total = total.saturating_add(size);
                if total > MAX_BYTES {
                    return Err(format!("bundle: more than {MAX_BYTES} bytes of content"));
                }
                let body = bytes
                    .get(at..at.saturating_add(size))
                    .ok_or("bundle: the archive ends inside a file")?;
                at += size.div_ceil(BLOCK) * BLOCK;
                let Some(rel) = rel else {
                    return Err(format!("bundle: unsafe entry {path}: not a file name"));
                };
                let target = dest.join(rel);
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("bundle: creating the directory of {path}: {e}"))?;
                }
                std::fs::File::create(&target)
                    .and_then(|mut f| f.write_all(body))
                    .map_err(|e| format!("bundle: writing {path}: {e}"))?;
            }
            b'5' => {
                if let Some(rel) = rel {
                    std::fs::create_dir_all(dest.join(rel))
                        .map_err(|e| format!("bundle: creating {path}: {e}"))?;
                }
            }
            b'x' | b'g' | b'L' | b'K' => {
                return Err(format!(
                    "bundle: unsafe entry {path}: pax and GNU extension headers are not accepted; \
                     build the bundle with `tar --format=ustar`"
                ));
            }
            other => {
                return Err(format!(
                    "bundle: unsafe entry {path}: type {:?} is not a regular file or directory",
                    char::from(other)
                ));
            }
        }
    }
    Err("bundle: the archive ends without its end marker".into())
}

/// The entry's path: `prefix/name` of a ustar header.
fn entry_path(header: &[u8]) -> Result<String, String> {
    let field = |range: std::ops::Range<usize>| {
        let raw = &header[range];
        let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
        std::str::from_utf8(&raw[..end]).map(str::to_owned)
    };
    let name = field(0..100).map_err(|_| "bundle: an entry name is not UTF-8")?;
    let ustar = &header[257..262] == b"ustar";
    let prefix = if ustar {
        field(345..500)
    } else {
        Ok(String::new())
    }
    .map_err(|_| "bundle: an entry name is not UTF-8")?;
    Ok(if prefix.is_empty() {
        name
    } else {
        format!("{prefix}/{name}")
    })
}

/// `path` as a relative path of plain components, `None` for the archive root (`./`). Anything that
/// could leave the destination is refused.
fn safe_relative(path: &str) -> Result<Option<PathBuf>, String> {
    if path.starts_with('/') || path.contains('\\') {
        return Err(format!("bundle: unsafe entry {path}: not a relative path"));
    }
    let mut rel = PathBuf::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => return Err(format!("bundle: unsafe entry {path}: climbs out with `..`")),
            part => rel.push(part),
        }
    }
    Ok((!rel.as_os_str().is_empty()).then_some(rel))
}

fn octal(field: &[u8]) -> Option<usize> {
    let text = std::str::from_utf8(field).ok()?;
    let text = text.trim_matches(|c| c == '\0' || c == ' ');
    usize::from_str_radix(text, 8).ok()
}

fn checksum_ok(header: &[u8]) -> bool {
    let sum: usize = header
        .iter()
        .enumerate()
        .map(|(i, b)| {
            if (148..156).contains(&i) {
                32
            } else {
                usize::from(*b)
            }
        })
        .sum();
    octal(&header[148..156]) == Some(sum)
}

/// A bundle directory seen as the grant's workspace: reads of paths under `workspace` are served from
/// `dir`, every other path does not exist, and nothing can be written or content-searched. What the
/// discovery loaders need (`stat`, `read_bytes`, `list_dir`, `glob`) and nothing more; the paths they report are the
/// workspace's, so the prompt names the paths a sandbox holding the same files would.
pub struct BundleFs {
    local: LocalFs,
    workspace: PathBuf,
    dir: PathBuf,
    /// Keeps the cache from evicting the directory while this filesystem can still be read.
    _pin: Option<Arc<Pin>>,
}

impl BundleFs {
    pub fn new(workspace: &Path, dir: &Path) -> Self {
        Self {
            local: LocalFs::new(),
            workspace: workspace.to_path_buf(),
            dir: dir.to_path_buf(),
            _pin: None,
        }
    }

    /// The pinned bundle as the workspace; it stays cached as long as this does.
    pub fn pinned(workspace: &Path, pin: &Arc<Pin>) -> Self {
        Self {
            _pin: Some(Arc::clone(pin)),
            ..Self::new(workspace, pin.dir())
        }
    }

    /// Where `path` lives in the bundle, if it is inside the workspace and climbs nowhere.
    fn inside(&self, path: &Path) -> Option<PathBuf> {
        let rel = path.strip_prefix(&self.workspace).ok()?;
        rel.components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
            .then(|| self.dir.join(rel))
    }
}

fn read_only() -> FsError {
    FsError::Backend("this session's files are a read-only bundle".into())
}

#[async_trait]
impl FsBackend for BundleFs {
    async fn search(&self, _q: &SearchQuery) -> Result<SearchOutcome, FsError> {
        Err(read_only())
    }

    async fn stat(&self, path: &Path) -> Result<Option<Meta>, FsError> {
        match self.inside(path) {
            Some(real) => self.local.stat(&real).await,
            None => Ok(None),
        }
    }

    async fn read_bytes(&self, path: &Path, offset: u64, max: usize) -> Result<Vec<u8>, FsError> {
        match self.inside(path) {
            Some(real) => self.local.read_bytes(&real, offset, max).await,
            None => Err(FsError::Backend(format!(
                "{}: no such file",
                path.display()
            ))),
        }
    }

    async fn write_bytes(&self, _path: &Path, _bytes: Vec<u8>) -> Result<(), FsError> {
        Err(read_only())
    }

    async fn write_if_unchanged(
        &self,
        _path: &Path,
        _bytes: Vec<u8>,
        _expected: Option<std::time::SystemTime>,
    ) -> Result<bool, FsError> {
        Err(read_only())
    }

    async fn create_dir_all(&self, _path: &Path) -> Result<(), FsError> {
        Err(read_only())
    }

    async fn list_dir(
        &self,
        path: &Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<DirEntry>, FsError> {
        match self.inside(path) {
            Some(real) => self.local.list_dir(&real, cap, include_hidden).await,
            None => Err(FsError::Backend(format!(
                "{}: no such directory",
                path.display()
            ))),
        }
    }

    fn world(&self) -> PathWorld {
        // Not this host's filesystem: paths resolve lexically, and `~` means nothing here.
        PathWorld::Remote { home: None }
    }

    async fn glob(&self, q: &GlobQuery) -> Result<GlobOutcome, FsError> {
        let Some(real) = self.inside(&q.root) else {
            return Ok(GlobOutcome::default());
        };
        let mut found = self
            .local
            .glob(&GlobQuery {
                root: real,
                ..q.clone()
            })
            .await?;
        // Report the workspace's spelling of each path, not the cache's.
        for (path, _) in &mut found.paths {
            if let Ok(rel) = path.strip_prefix(&self.dir) {
                *path = self.workspace.join(rel);
            }
        }
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use beyond_test_support::bundle::{BundleServer, Entry, Gauge, dir, file, sha256_hex, tar};

    use super::*;

    fn unpacked(bytes: &[u8]) -> Result<tempfile::TempDir, String> {
        let out = tempfile::tempdir().unwrap();
        unpack(bytes, out.path()).map(|_| out)
    }

    fn special(path: &str, kind: u8, link: &str) -> Vec<u8> {
        tar(&[Entry {
            path,
            kind,
            body: b"",
            link,
        }])
    }

    #[test]
    fn files_and_directories_unpack_where_the_archive_puts_them() {
        let out = unpacked(&tar(&[
            dir("./"),
            dir("./.agents/"),
            file("./AGENTS.md", "hello"),
            file(".agents/skills/s/SKILL.md", "skill"),
        ]))
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(out.path().join("AGENTS.md")).unwrap(),
            "hello"
        );
        assert_eq!(
            std::fs::read_to_string(out.path().join(".agents/skills/s/SKILL.md")).unwrap(),
            "skill"
        );
    }

    #[test]
    fn an_entry_that_could_leave_the_directory_or_is_not_a_file_refuses_the_bundle() {
        for (name, bytes) in [
            ("parent", tar(&[file("../x", "x")])),
            ("nested parent", tar(&[file("a/../../x", "x")])),
            ("absolute", tar(&[file("/etc/x", "x")])),
            ("backslash", tar(&[file("a\\..\\x", "x")])),
            ("symlink", special("l", b'2', "/etc/passwd")),
            ("hardlink", special("l", b'1', "AGENTS.md")),
            ("character device", special("d", b'3', "")),
            ("block device", special("d", b'4', "")),
            ("fifo", special("d", b'6', "")),
            ("pax header", special("p", b'x', "")),
            ("gnu long name", special("p", b'L', "")),
        ] {
            let err = unpacked(&bytes).unwrap_err();
            assert!(err.contains("unsafe entry"), "{name}: {err}");
        }
        for kind in [b'x', b'g', b'L', b'K'] {
            let err = unpacked(&special("p", kind, "")).unwrap_err();
            assert!(err.contains("tar --format=ustar"), "{err}");
        }
    }

    #[test]
    fn a_damaged_archive_refuses_the_bundle() {
        let mut bytes = tar(&[file("AGENTS.md", "hello")]);
        bytes[0] = b'B'; // the name changes under the checksum
        assert!(unpacked(&bytes).unwrap_err().contains("corrupt"));
        let whole = tar(&[file("AGENTS.md", "hello")]);
        assert!(unpacked(&whole[..BLOCK + 2]).unwrap_err().contains("ends"));
        assert!(unpacked(&whole[..BLOCK]).unwrap_err().contains("ends"));
    }

    #[test]
    fn a_failed_unpack_leaves_nothing_in_the_cache() {
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("abc");
        install(root.path(), &dest, &special("l", b'2', "/etc")).unwrap_err();
        assert!(!dest.exists());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    fn egress() -> crate::tools::mcp::McpEgress {
        crate::tools::mcp::McpEgress::new(crate::tools::web::ssrf::EgressPolicy::new(true, &[]))
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sessions_opening_together_on_a_cold_cache_fetch_the_bundle_once() {
        let bytes = tar(&[file("AGENTS.md", "hello")]);
        let server = BundleServer::start(bytes.clone());
        let bundle = Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(&bytes),
        };
        let root = tempfile::tempdir().unwrap();
        let egress = egress();
        let cache = Cache::new(root.path(), u64::MAX);
        let opens = (0..16).map(|_| cache.ensure(&bundle, Some(&egress)));
        for pin in futures::future::join_all(opens).await {
            assert!(pin.unwrap().dir().join("AGENTS.md").is_file());
        }
        assert_eq!(server.hits(), 1, "one fetch for one hash");
    }

    /// Opening a session on a bundle that is already cached checks the disk from the blocking pool,
    /// never from the async thread that every other session shares.
    #[test]
    fn a_cached_bundle_is_checked_off_the_runtime_thread() {
        let bytes = tar(&[file("AGENTS.md", "hello")]);
        let server = BundleServer::start(bytes.clone());
        let bundle = Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(&bytes),
        };
        let root = tempfile::tempdir().unwrap();
        let cache = Cache::new(root.path(), u64::MAX);
        let warm = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        warm.block_on(cache.ensure(&bundle, Some(&egress())))
            .unwrap();
        // A current-thread runtime starts threads only for blocking work, so a hit that starts none
        // did its `stat` on the async thread.
        let started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&started);
        let hit = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .on_thread_start(move || {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })
            .build()
            .unwrap();
        hit.block_on(cache.ensure(&bundle, None)).unwrap();
        assert_eq!(started.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failed_fetch_does_not_poison_the_next_attempt() {
        let bytes = tar(&[file("AGENTS.md", "hello")]);
        let server = BundleServer::start(bytes.clone());
        let mut bundle = Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(&bytes),
        };
        let root = tempfile::tempdir().unwrap();
        let egress = egress();
        let good = bundle.url.clone();
        bundle.url = "http://127.0.0.1:1/agent.tar".into();
        let cache = Cache::new(root.path(), u64::MAX);
        assert!(cache.ensure(&bundle, Some(&egress)).await.is_err());
        bundle.url = good;
        cache.ensure(&bundle, Some(&egress)).await.unwrap();
    }

    /// The disk one of the `served` bundles occupies once unpacked.
    fn served_disk() -> u64 {
        let out = unpacked(&tar(&[file("AGENTS.md", "bundle-000")])).unwrap();
        disk_bytes(out.path())
    }

    /// A ten-byte bundle (distinct per `n`) served from its own loopback server.
    fn served(n: u8) -> (BundleServer, Bundle) {
        let bytes = tar(&[file("AGENTS.md", &format!("bundle-{n:03}"))]);
        let server = BundleServer::start(bytes.clone());
        let bundle = Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(&bytes),
        };
        (server, bundle)
    }

    /// What `du` reports for `dir`: allocated blocks of every file and directory in it.
    fn du_bytes(dir: &Path) -> u64 {
        use std::os::unix::fs::MetadataExt as _;
        let mut total = std::fs::symlink_metadata(dir).unwrap().blocks() * 512;
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let meta = e.metadata().unwrap();
            total += if meta.is_dir() {
                du_bytes(&e.path())
            } else {
                meta.blocks() * 512
            };
        }
        total
    }

    #[tokio::test]
    async fn many_tiny_bundles_are_held_to_the_budget_by_the_disk_they_occupy() {
        let root = tempfile::tempdir().unwrap();
        let egress = egress();
        // A ten-byte bundle occupies whole blocks for its file and directory: room for a few on disk.
        let budget = 3 * served_disk();
        let cache = Cache::new(root.path(), budget);
        let mut servers = Vec::new();
        for n in 0..40 {
            let (server, bundle) = served(n);
            drop(cache.ensure(&bundle, Some(&egress)).await.unwrap());
            servers.push(server);
        }
        let disk = du_bytes(cache.dir());
        assert!(
            disk <= budget + served_disk(),
            "{disk} bytes on disk against a budget of {budget}"
        );
    }

    #[tokio::test]
    async fn over_budget_evicts_the_oldest_unpinned_bundle_and_never_a_pinned_one() {
        let root = tempfile::tempdir().unwrap();
        let egress = egress();
        // Room for two of the ten-byte bundles.
        let cache = Cache::new(root.path(), served_disk() * 5 / 2);
        let ((_a, a), (_b, b), (_c, c)) = (served(1), served(2), served(3));
        let held_oldest = cache.ensure(&a, Some(&egress)).await.unwrap();
        drop(cache.ensure(&b, Some(&egress)).await.unwrap());
        drop(cache.ensure(&c, Some(&egress)).await.unwrap());
        assert!(cache.dir().join(&a.sha256).is_dir(), "pinned: kept");
        assert!(
            !cache.dir().join(&b.sha256).exists(),
            "oldest unpinned: evicted"
        );
        assert!(cache.dir().join(&c.sha256).is_dir());
        assert!(held_oldest.dir().join("AGENTS.md").is_file());
        let names: Vec<_> = std::fs::read_dir(cache.dir())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|n| n != LOCK)
            .collect();
        assert_eq!(names.len(), 2, "no .trash or .unpack leftovers: {names:?}");
    }

    #[tokio::test]
    async fn bundles_released_while_over_budget_are_trimmed_without_waiting_for_the_next_install() {
        let root = tempfile::tempdir().unwrap();
        let egress = egress();
        // Room for one ten-byte bundle; two sessions hold one each, so both stay while they are held.
        let cache = Cache::new(root.path(), served_disk() * 3 / 2);
        let ((_a, a), (_b, b)) = (served(1), served(2));
        let held_a = cache.ensure(&a, Some(&egress)).await.unwrap();
        let held_b = cache.ensure(&b, Some(&egress)).await.unwrap();
        assert!(held_a.dir().is_dir() && held_b.dir().is_dir());
        // Both sessions end and nothing else arrives: the cache must still come back within budget.
        drop(held_a);
        drop(held_b);
        let kept = [&a, &b]
            .iter()
            .filter(|bundle| cache.dir().join(&bundle.sha256).is_dir())
            .count();
        assert_eq!(
            kept, 1,
            "over budget with nothing held: the older is evicted"
        );
        assert!(
            !cache.dir().join(&a.sha256).exists(),
            "the least recently released goes first"
        );
    }

    #[tokio::test]
    async fn a_bundle_requested_after_its_eviction_is_fetched_again() {
        let root = tempfile::tempdir().unwrap();
        let egress = egress();
        let cache = Cache::new(root.path(), served_disk() * 3 / 2);
        let ((sa, a), (_b, b)) = (served(1), served(2));
        drop(cache.ensure(&a, Some(&egress)).await.unwrap());
        drop(cache.ensure(&b, Some(&egress)).await.unwrap());
        assert!(!cache.dir().join(&a.sha256).exists());
        let again = cache.ensure(&a, Some(&egress)).await.unwrap();
        assert_eq!(sa.hits(), 2, "evicted, so fetched again");
        assert!(again.dir().join("AGENTS.md").is_file());
    }

    #[tokio::test]
    async fn caches_sharing_a_base_never_touch_each_others_directories() {
        let base = tempfile::tempdir().unwrap();
        let egress = egress();
        let (_a, a) = served(1);
        let theirs = Cache::new(base.path(), u64::MAX);
        let held = theirs.ensure(&a, Some(&egress)).await.unwrap();
        std::fs::create_dir(theirs.dir().join(".unpack-in-progress")).unwrap();
        // A second process starts and, with a budget nothing fits in, evicts all it can.
        let ours = Cache::new(base.path(), 0);
        let ((_b, b), (_c, c)) = (served(2), served(3));
        drop(ours.ensure(&b, Some(&egress)).await.unwrap());
        drop(ours.ensure(&c, Some(&egress)).await.unwrap());
        assert!(
            !ours.dir().join(&b.sha256).exists(),
            "its own entry is evicted"
        );
        assert!(
            held.dir().join("AGENTS.md").is_file(),
            "the other's pinned dir stays"
        );
        assert!(theirs.dir().join(".unpack-in-progress").is_dir());
    }

    #[tokio::test]
    async fn a_dead_owners_directory_is_swept_by_the_next_start() {
        let base = tempfile::tempdir().unwrap();
        let egress = egress();
        let (_a, a) = served(1);
        let dead = Cache::new(base.path(), u64::MAX);
        let left = dead.dir().to_path_buf();
        drop(dead.ensure(&a, Some(&egress)).await.unwrap());
        std::fs::create_dir(left.join(".unpack-9-9")).unwrap();
        // Dropping the cache closes its lock file, as a crashed process's exit does.
        drop(dead);
        assert!(left.is_dir());
        let next = Cache::new(base.path(), u64::MAX);
        assert!(!left.exists(), "swept");
        assert!(next.dir().is_dir());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distinct_bundles_are_fetched_no_more_than_the_cap_at_a_time() {
        let gauge = Arc::new(Gauge::default());
        let egress = egress();
        let root = tempfile::tempdir().unwrap();
        let cache = Cache::with_fetch_limit(root.path(), u64::MAX, 1);
        let servers: Vec<_> = (0..4)
            .map(|n| {
                let bytes = tar(&[file("AGENTS.md", &format!("bundle {n}"))]);
                let held = std::time::Duration::from_millis(100);
                let server = BundleServer::start_held(bytes.clone(), gauge.clone(), held);
                let bundle = Bundle {
                    url: server.url.clone(),
                    sha256: sha256_hex(&bytes),
                };
                (server, bundle)
            })
            .collect();
        let opens = servers.iter().map(|(_, b)| cache.ensure(b, Some(&egress)));
        for pin in futures::future::join_all(opens).await {
            assert!(pin.is_ok());
        }
        assert_eq!(gauge.peak(), 1, "one fetch at a time under a cap of one");
    }

    #[tokio::test]
    async fn the_bundle_is_read_at_the_workspace_and_nowhere_else() {
        let out = unpacked(&tar(&[file("AGENTS.md", "hello")])).unwrap();
        let fs = BundleFs::new(Path::new("/workspace"), out.path());
        assert!(
            fs.stat(Path::new("/workspace/AGENTS.md"))
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            fs.stat(Path::new("/workspace/nope"))
                .await
                .unwrap()
                .is_none()
        );
        assert!(fs.stat(Path::new("/AGENTS.md")).await.unwrap().is_none());
        assert!(
            fs.stat(Path::new("/workspace/../x"))
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            fs.read_bytes(Path::new("/workspace/AGENTS.md"), 0, 100)
                .await
                .unwrap(),
            b"hello"
        );
        assert!(
            fs.read_bytes(Path::new("/etc/passwd"), 0, 100)
                .await
                .is_err()
        );
        assert!(
            fs.write_bytes(Path::new("/workspace/x"), vec![])
                .await
                .is_err()
        );
    }
}
