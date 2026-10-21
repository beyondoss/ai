//! The agent as a deployment: a grant's [`Bundle`] is an immutable plain tar of the agent folder
//! (`AGENTS.md`, `.mcp.json`, `.agents/skills`, `.agents/agents`, ...), named by the SHA-256 of its
//! bytes the way an image is named by its digest.
//!
//! [`ensure`] fetches it through the replica's egress client, checks the hash **before** a byte is
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

/// Where bundles unpack: `agent-bundles` under the replica's temp dir (`TMPDIR`). A cache only: a
/// missing directory is fetched again.
pub fn cache_root() -> PathBuf {
    std::env::temp_dir().join("agent-bundles")
}

/// How many bundles this process fetches at once. Each body is buffered whole (up to
/// [`MAX_BYTES`]), so this bounds the peak at about 256 MiB however many tenants cold-start at once.
const CONCURRENT_FETCHES: usize = 4;

/// The unpacked bundle under `root`, fetching, verifying and unpacking it first if it is not there.
pub async fn ensure(
    root: &Path,
    bundle: &Bundle,
    egress: Option<&crate::tools::mcp::McpEgress>,
) -> Result<PathBuf, String> {
    static FETCHES: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(CONCURRENT_FETCHES);
    ensure_limited(root, bundle, egress, &FETCHES).await
}

async fn ensure_limited(
    root: &Path,
    bundle: &Bundle,
    egress: Option<&crate::tools::mcp::McpEgress>,
    fetches: &tokio::sync::Semaphore,
) -> Result<PathBuf, String> {
    if !bundle.is_well_formed() {
        return Err("bundle: the grant's bundle is not a url and a lowercase sha256".into());
    }
    let dest = root.join(&bundle.sha256);
    if dest.is_dir() {
        return Ok(dest);
    }
    let egress = egress.ok_or("bundle: this replica has no egress client to fetch it with")?;
    // One fetch per hash in flight in this process: the others wait here and find the unpacked copy.
    // A failed attempt just releases the lock, so the next waiter tries for itself.
    let flight = flight(&dest);
    let turn = flight.lock().await;
    let result = fetch_and_install(root, &dest, bundle, egress, fetches).await;
    drop(turn);
    release(&dest, flight);
    result
}

/// The lock for fetching the bundle that unpacks to `dest`.
fn flight(dest: &Path) -> Arc<tokio::sync::Mutex<()>> {
    FLIGHTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry(dest.to_path_buf())
        .or_default()
        .clone()
}

/// Forget `dest`'s lock once nobody else holds or waits on it.
fn release(dest: &Path, flight: Arc<tokio::sync::Mutex<()>>) {
    let mut flights = FLIGHTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // The map's handle and ours: no waiter has cloned it.
    if Arc::strong_count(&flight) == 2 {
        flights.remove(dest);
    }
}

type Flights = std::sync::Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>;

static FLIGHTS: std::sync::LazyLock<Flights> = std::sync::LazyLock::new(Default::default);

async fn fetch_and_install(
    root: &Path,
    dest: &Path,
    bundle: &Bundle,
    egress: &crate::tools::mcp::McpEgress,
    fetches: &tokio::sync::Semaphore,
) -> Result<PathBuf, String> {
    if dest.is_dir() {
        return Ok(dest.to_path_buf());
    }
    let _permit = fetches.acquire().await.map_err(|e| e.to_string())?;
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
    let (root, dest) = (root.to_path_buf(), dest.to_path_buf());
    tokio::task::spawn_blocking(move || install(&root, &dest, &bytes))
        .await
        .map_err(|e| format!("bundle: unpacking failed: {e}"))?
}

/// Unpack `bytes` beside `dest` and rename it into place. Losing the rename to another session that
/// unpacked the same bundle is success: the content is named by its hash.
fn install(root: &Path, dest: &Path, bytes: &[u8]) -> Result<PathBuf, String> {
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
    match std::fs::rename(&tmp, dest) {
        Ok(()) => Ok(dest.to_path_buf()),
        Err(_) if dest.is_dir() => {
            let _ = std::fs::remove_dir_all(&tmp);
            Ok(dest.to_path_buf())
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            Err(io("installing it", e))
        }
    }
}

/// Unpack a plain tar into the existing, empty directory `dest`.
pub fn unpack(bytes: &[u8], dest: &Path) -> Result<(), String> {
    let mut at = 0;
    let mut total = 0usize;
    while let Some(header) = bytes.get(at..at + BLOCK) {
        at += BLOCK;
        if header.iter().all(|b| *b == 0) {
            return Ok(());
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
}

impl BundleFs {
    pub fn new(workspace: &Path, dir: &Path) -> Self {
        Self {
            local: LocalFs::new(),
            workspace: workspace.to_path_buf(),
            dir: dir.to_path_buf(),
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

    use beyond_ai_test_support::bundle::{BundleServer, Entry, Gauge, dir, file, sha256_hex, tar};

    use super::*;

    fn unpacked(bytes: &[u8]) -> Result<tempfile::TempDir, String> {
        let out = tempfile::tempdir().unwrap();
        unpack(bytes, out.path()).map(|()| out)
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
    fn a_cached_bundle_is_reused_and_a_racing_install_is_not_an_error() {
        let root = tempfile::tempdir().unwrap();
        let bytes = tar(&[file("AGENTS.md", "hello")]);
        let dest = root.path().join("abc");
        install(root.path(), &dest, &bytes).unwrap();
        // The same hash unpacked again by a second session: the first copy stays.
        install(root.path(), &dest, &bytes).unwrap();
        assert_eq!(
            std::fs::read_to_string(dest.join("AGENTS.md")).unwrap(),
            "hello"
        );
        let leftovers: Vec<_> = std::fs::read_dir(root.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(leftovers.len(), 1, "{leftovers:?}");
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
        let opens = (0..16).map(|_| ensure(root.path(), &bundle, Some(&egress)));
        for dir in futures::future::join_all(opens).await {
            assert!(dir.unwrap().join("AGENTS.md").is_file());
        }
        assert_eq!(server.hits(), 1, "one fetch for one hash");
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
        ensure(root.path(), &bundle, Some(&egress))
            .await
            .unwrap_err();
        bundle.url = good;
        ensure(root.path(), &bundle, Some(&egress)).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distinct_bundles_are_fetched_no_more_than_the_cap_at_a_time() {
        let gauge = Arc::new(Gauge::default());
        let egress = egress();
        let root = tempfile::tempdir().unwrap();
        let limit = tokio::sync::Semaphore::new(1);
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
        let opens = servers
            .iter()
            .map(|(_, b)| ensure_limited(root.path(), b, Some(&egress), &limit));
        for dir in futures::future::join_all(opens).await {
            dir.unwrap();
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
