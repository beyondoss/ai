//! Stateful property test for `ShellFs`, checked against ground truth.
//!
//! The oracle never asks another `FsBackend`. Every expectation comes from `std::fs` reading the same
//! root (see `truth`, `search_truth`) and from FILES-OVER-EXEC (sdk/conformance), so a defect that
//! `ShellFs` and `LocalFs` share is still caught. The transport is a real loopback execd.
//!
//! Operations are generated sequences over a small tree: `ShellFs` calls (write, write_if_unchanged,
//! read, stat, list, search, mkdir) interleaved with **environment** steps done directly on disk
//! (write, mkdir, symlink, chmod, remove, touch, a file whose name is not UTF-8). See `Config` for
//! the environments and `faults` for the fault model.
//!
//! Reproduce: the CI run uses fixed seeds and case counts (`settings`). `SHELLFS_PROPS_CASES=N` and
//! `SHELLFS_PROPS_SEED=S` widen or move the search locally. A failing case is shrunk by Hegel and its
//! steps are printed (`tc.note`) in order.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod faults;
pub mod fixtures;
pub mod search_truth;
pub mod truth;

use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use beyond_agent::exec_endpoint::HttpExecRunner;
use beyond_agent::tools::exec::CommandRunner;
use beyond_agent::tools::fs::shell::{Capabilities, SearchEngine, ShellFs};
use beyond_agent::tools::fs::{FileKind, FsBackend, FsError, SearchQuery};
use beyond_test_support::exec_mock::{ExecMock, MockOptions};
use hegel::TestCase;
use hegel::generators as gs;

use faults::FaultRunner;
use search_truth::{Engine, Spec, Th};
use truth::{Kind, Node, Plan, Snap, classify, secs, snapshot, temps};

/// One environment the machine runs in.
#[derive(Clone, Copy)]
pub struct Config {
    /// The runner's response cap. `None` is the default 16 MiB, which no generated file reaches.
    pub cap: Option<usize>,
    /// Whether writes use stdin (one command) or 48 KiB argv chunks (the rung for a v1 endpoint).
    pub stdin: bool,
    /// Percent of requests a fault hits. 0 is a clean transport.
    pub fault_pct: u64,
    /// Start with a read-only directory and read-only files in the tree.
    pub readonly_start: bool,
    /// Run in a user+mount namespace whose root is a tmpfs of this many KiB, so writes can hit a real
    /// ENOSPC. Needs `unshare` and `nsenter` and no privilege; the test skips, with the reason
    /// printed, where they are unavailable. Commands reach it through `nsenter`, not execd.
    pub tiny_kib: Option<usize>,
    pub steps: i64,
}

pub const PLAIN: Config = Config {
    cap: None,
    stdin: true,
    fault_pct: 0,
    readonly_start: false,
    tiny_kib: None,
    steps: 30,
};

/// Fixed seed and case count for CI; the environment widens them locally.
pub fn settings(default_cases: u64, seed: u64) -> hegel::Settings {
    let cases = std::env::var("SHELLFS_PROPS_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_cases);
    let seed = std::env::var("SHELLFS_PROPS_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(seed);
    hegel::Settings::new()
        .test_cases(cases)
        .seed(Some(seed))
        .database(None)
        .show_statistics(std::env::var_os("SHELLFS_PROPS_STATS").is_some())
        .suppress_health_check(hegel::HealthCheck::all())
}

pub fn truth_is_root() -> bool {
    truth::is_root()
}

pub fn rt() -> &'static tokio::runtime::Runtime {
    static R: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    R.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    })
}

/// One loopback execd (the real endpoint behind `ExecMock`'s conforming configuration) per process.
fn endpoint() -> &'static str {
    static U: OnceLock<String> = OnceLock::new();
    U.get_or_init(|| {
        rt().block_on(async {
            let root: &'static tempfile::TempDir =
                Box::leak(Box::new(tempfile::tempdir().unwrap()));
            ExecMock::start_with(root.path(), MockOptions::default())
                .await
                .url
        })
    })
}

const DEFAULT_CAP: usize = 16 * 1024 * 1024;

pub fn http_runner(cap: Option<usize>) -> Arc<HttpExecRunner> {
    static M: OnceLock<Mutex<HashMap<usize, Arc<HttpExecRunner>>>> = OnceLock::new();
    let cap = cap.unwrap_or(DEFAULT_CAP);
    M.get_or_init(Mutex::default)
        .lock()
        .unwrap()
        .entry(cap)
        .or_insert_with(|| {
            Arc::new(
                HttpExecRunner::new(endpoint())
                    .unwrap()
                    .with_max_response_bytes(cap),
            )
        })
        .clone()
}

/// `ShellFs::connect` returns the backend on `main` and a `Result` once a failed probe fails connect
/// (D3). This test compiles against both.
pub trait IntoFs {
    fn into_fs(self) -> Option<ShellFs>;
}
impl IntoFs for ShellFs {
    fn into_fs(self) -> Option<ShellFs> {
        Some(self)
    }
}
impl<E> IntoFs for Result<ShellFs, E> {
    fn into_fs(self) -> Option<ShellFs> {
        self.ok()
    }
}

pub async fn connect(r: Arc<dyn CommandRunner>) -> Option<ShellFs> {
    ShellFs::connect(r).await.into_fs()
}

fn capabilities(cap: Option<usize>) -> Capabilities {
    static C: OnceLock<Mutex<HashMap<usize, Capabilities>>> = OnceLock::new();
    let key = cap.unwrap_or(DEFAULT_CAP);
    if let Some(c) = C.get_or_init(Mutex::default).lock().unwrap().get(&key) {
        return *c;
    }
    let fs = rt()
        .block_on(connect(http_runner(cap)))
        .expect("the loopback execd answers the probe");
    let caps = fs.capabilities();
    C.get().unwrap().lock().unwrap().insert(key, caps);
    caps
}

pub struct Machine {
    cfg: Config,
    /// Where `std::fs` (the oracle) sees the tree.
    root: PathBuf,
    /// Where the commands `ShellFs` runs see it: the same unless the tree is in a namespace.
    cmd_root: PathBuf,
    _tiny: Option<Tiny>,
    dir: Option<tempfile::TempDir>,
    fs: ShellFs,
    fs2: ShellFs,
    faults: Arc<FaultRunner>,
    engine: Engine,
    known: Vec<PathBuf>,
    stats: HashMap<PathBuf, Option<u64>>,
}

impl Drop for Machine {
    fn drop(&mut self) {
        // Unlock whatever the run chmodded, so the temp dir can be deleted.
        let _ = snapshot(&self.root);
        if let Some(d) = self.dir.take() {
            drop(d);
        }
    }
}

/// A tmpfs of a few KiB in a private user+mount namespace, held open by a `sleep`. Commands run in it
/// through `nsenter`; the test reads it through `/proc/PID/root`. No privilege needed.
pub struct Tiny {
    child: std::process::Child,
    pub pid: u32,
}

impl Drop for Tiny {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn start_tiny(mount: &Path, kib: usize) -> Result<Tiny, String> {
    use std::io::BufRead;
    let mut child = std::process::Command::new("unshare")
        .args(["-rm", "sh", "-c"])
        .arg(format!(
            "mount -t tmpfs -o size={kib}k tmpfs '{}' && echo ready && exec sleep 100000",
            mount.display()
        ))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run unshare: {e}"))?;
    let mut line = String::new();
    std::io::BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    if line.trim() != "ready" {
        let mut err = String::new();
        let _ = std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut err);
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!(
            "an unprivileged tmpfs mount is not available: {}",
            err.trim()
        ));
    }
    let pid = child.id();
    Ok(Tiny { child, pid })
}

/// Why the tiny-filesystem environment cannot run here, or `None` when it can.
pub fn tiny_unavailable() -> Option<String> {
    let dir = tempfile::tempdir().unwrap();
    match start_tiny(dir.path(), 64) {
        Err(e) => Some(e),
        Ok(t) => {
            let ok = std::process::Command::new("nsenter")
                .args([
                    "-t",
                    &t.pid.to_string(),
                    "-U",
                    "-m",
                    "--preserve-credentials",
                    "--",
                    "true",
                ])
                .status()
                .is_ok_and(|s| s.success());
            (!ok).then(|| "nsenter cannot join the namespace".to_string())
        }
    }
}

/// Runs each command inside the tiny filesystem's namespace.
struct NsRunner {
    pid: u32,
}

#[async_trait::async_trait]
impl CommandRunner for NsRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        self.go(program, args, cwd, timeout, None).await
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        self.go(program, args, cwd, timeout, Some(stdin)).await
    }
}

impl NsRunner {
    async fn go(
        &self,
        program: &str,
        args: &[String],
        _cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<beyond_agent::tools::exec::ExecResult> {
        use tokio::io::AsyncWriteExt;
        let mut c = tokio::process::Command::new("nsenter");
        c.args([
            "-t",
            &self.pid.to_string(),
            "-U",
            "-m",
            "--preserve-credentials",
            "--",
        ])
        .arg(program)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
        let mut child = c.spawn()?;
        let mut sin = child.stdin.take().unwrap();
        let data = stdin.map(<[u8]>::to_vec);
        let feed = async move {
            if let Some(d) = data {
                let _ = sin.write_all(&d).await;
            }
            drop(sin);
        };
        let (out, ()) = tokio::time::timeout(timeout, async {
            tokio::join!(child.wait_with_output(), feed)
        })
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "timed out"))?;
        let out = out?;
        Ok(beyond_agent::tools::exec::ExecResult {
            code: out.status.code(),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            ..Default::default()
        })
    }
}

pub fn machine(tc: &TestCase, cfg: Config) -> Machine {
    let dir = tempfile::tempdir().unwrap();
    let host_root = dir.path().canonicalize().unwrap();
    let (root, tiny) = match cfg.tiny_kib {
        Some(kib) => {
            let t = start_tiny(&host_root, kib).expect("checked by tiny_unavailable");
            (
                PathBuf::from(format!("/proc/{}/root{}", t.pid, host_root.display())),
                Some(t),
            )
        }
        None => (host_root.clone(), None),
    };
    let seed = tc.draw_silent(gs::integers::<u64>());
    let mut caps = capabilities(cfg.cap);
    caps.stdin &= cfg.stdin;
    let http: Arc<dyn CommandRunner> = match &tiny {
        Some(t) => Arc::new(NsRunner { pid: t.pid }),
        None => http_runner(cfg.cap),
    };
    let faults = FaultRunner::new(http.clone(), seed, cfg.fault_pct);
    let fs = ShellFs::with_capabilities(faults.clone(), caps);
    let fs2 = ShellFs::with_capabilities(http, caps);
    faults.enable(cfg.fault_pct > 0);
    let engine = match caps.search_engine() {
        SearchEngine::Ripgrep => Engine::Rg,
        _ => Engine::Grep,
    };
    let mut m = Machine {
        cfg,
        root: root.clone(),
        cmd_root: host_root,
        _tiny: tiny,
        dir: Some(dir),
        fs,
        fs2,
        faults,
        engine,
        known: vec![],
        stats: HashMap::new(),
    };
    // A small tree to start from, so reads, listings and searches have something to find.
    let seed: &[(&str, &[u8])] = &[
        ("alpha.txt", b"alpha\nneedle one\nbeta\nNeedle two\n"),
        ("d/n.txt", b"foo bar\nneedle\r\nlast line no newline"),
        ("d/sub/x", b"x1\ne1\n\nneedle\n"),
        (".hidden", b"needle hidden\n"),
    ];
    for (rel, data) in seed {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, data).unwrap();
        m.known.push(p);
    }
    m.known.extend([root.join("d"), root.join("d/sub")]);
    if cfg.readonly_start {
        let ro = root.join("ro");
        std::fs::create_dir(&ro).unwrap();
        std::fs::write(ro.join("kept.txt"), b"needle in ro\n").unwrap();
        std::fs::write(root.join("locked.txt"), b"needle locked\n").unwrap();
        std::fs::set_permissions(
            root.join("locked.txt"),
            std::fs::Permissions::from_mode(0o444),
        )
        .unwrap();
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).unwrap();
        m.known
            .extend([ro.clone(), ro.join("kept.txt"), root.join("locked.txt")]);
    }
    m
}

/// An open bug this test found: tolerated (and counted, and printed) until its fix lands, then removed
/// from here. `known_bugs.rs` holds one deterministic test per entry that fails once the bug is fixed,
/// so a fix cannot leave a stale exemption behind.
pub fn known(id: &str, msg: String) {
    eprintln!("KNOWN OPEN BUG {id}: {msg}");
}

#[track_caller]
fn fail(msg: impl AsRef<str>) -> ! {
    panic!("{}", msg.as_ref())
}

fn node_desc(n: &Node) -> String {
    match n {
        Node::File { mode, bytes, mtime } => format!(
            "file mode={mode:o} len={} mtime={} head={:?}",
            bytes.len(),
            secs(*mtime),
            &bytes[..bytes.len().min(24)]
        ),
        Node::Dir { mode, kids } => format!("dir mode={mode:o} kids={kids:?}"),
        Node::Link(t) => format!("symlink -> {t:?}"),
        Node::Other { mode } => format!("other mode={mode:o}"),
    }
}

/// What differs between two snapshots, one line per path.
fn diff(actual: &Snap, expected: &Snap, root: &Path) -> Vec<String> {
    let mut out = vec![];
    let rel = |p: &Path| p.strip_prefix(root).unwrap_or(p).display().to_string();
    for (p, a) in actual {
        match expected.get(p) {
            None => out.push(format!("unexpected {:?}: {}", rel(p), node_desc(a))),
            Some(e) if e != a => out.push(format!(
                "{:?}: actual {} / expected {}",
                rel(p),
                node_desc(a),
                node_desc(e)
            )),
            _ => {}
        }
    }
    for (p, e) in expected {
        if !actual.contains_key(p) {
            out.push(format!("missing {:?}: {}", rel(p), node_desc(e)));
        }
    }
    out
}

fn same_ignoring_mtimes(actual: &Snap, expected: &Snap, ignore: &[&Path]) -> Option<Snap> {
    // Returns the expected snapshot with the ignored files' mtimes (and a new file's mode) taken
    // from `actual`.
    let mut e = expected.clone();
    for p in ignore {
        if let (
            Some(Node::File { mtime, mode, .. }),
            Some(Node::File {
                mtime: am,
                mode: amode,
                ..
            }),
        ) = (e.get_mut(*p), actual.get(*p))
        {
            *mtime = *am;
            if *mode == 0o644 && *amode & 0o7000 == 0 && amode & 0o600 == 0o600 {
                *mode = *amode; // a new file's mode is the umask's business
            }
        }
    }
    Some(e)
}

fn read_only_ops_unchanged(m: &Machine, before: &Snap, what: &str) {
    let after = snapshot(&m.root);
    if before != &after {
        fail(format!(
            "{what} changed the tree:\n{}",
            diff(&after, before, &m.root).join("\n")
        ));
    }
}

impl Machine {
    fn block<F: std::future::Future>(&self, f: F) -> F::Output {
        rt().block_on(f)
    }

    /// The path as the commands see it.
    fn c(&self, p: &Path) -> PathBuf {
        self.cmd_root.join(p.strip_prefix(&self.root).unwrap_or(p))
    }

    fn fill(&self) -> Option<usize> {
        self.cfg.tiny_kib.map(|k| k * 1024)
    }

    /// Free bytes on the tree's filesystem, when it is the small one.
    fn free_bytes(&self) -> Option<u64> {
        self.cfg.tiny_kib?;
        let s = rustix::fs::statvfs(&self.root).ok()?;
        Some(s.f_bavail * s.f_frsize)
    }

    /// The disk-full rule: a temp of `n` bytes needs its blocks while the old file still holds its
    /// own, so a write that does not fit must fail (and clean up), one that clearly fits must
    /// succeed, and the last couple of blocks are the filesystem's business.
    fn with_space(&self, plan: Plan, n: usize) -> Plan {
        let (Some(free), Plan::Succeed) = (self.free_bytes(), &plan) else {
            return plan;
        };
        let need = (n as u64).div_ceil(4096);
        let have = free / 4096;
        if need > have {
            Plan::Fail(vec![Kind::Io])
        } else if need + 2 > have {
            Plan::Either
        } else {
            plan
        }
    }

    fn kind(&self, e: &FsError, p: &Path) -> Kind {
        classify(e, &[p, &self.c(p), &self.root, &self.cmd_root])
    }

    fn remember(&mut self, p: &Path) {
        if !self.known.contains(&p.to_path_buf()) {
            self.known.push(p.to_path_buf());
        }
    }

    fn pick_path(&mut self, tc: &TestCase) -> PathBuf {
        let existing = tc.draw(gs::integers::<u8>().max_value(9)) < 6 && !self.known.is_empty();
        let p = if existing {
            tc.draw(gs::sampled_from(self.known.clone()))
        } else {
            let mut parents = vec![self.root.clone(); 4];
            parents.push(self.root.join("missing"));
            let mut files = vec![];
            for k in &self.known {
                match std::fs::metadata(k) {
                    Ok(m) if m.is_dir() => parents.extend([k.clone(), k.clone()]),
                    Ok(_) => files.push(k.clone()),
                    Err(_) => {}
                }
            }
            if !files.is_empty() && tc.draw(gs::integers::<u8>().max_value(4)) == 0 {
                parents = files; // a file as the parent: the not-a-directory cases
            }
            let parent = tc.draw(gs::sampled_from(parents));
            parent.join(fixtures::name(tc))
        };
        self.remember(&p);
        p
    }

    /// Whether `target`, as a symlink at `link`, resolves (lexically, `..` popping a component) to a
    /// path inside the test root.
    fn stays_inside(&self, link: &Path, target: &Path) -> bool {
        use std::path::Component;
        let Ok(root) = std::fs::canonicalize(&self.root) else {
            return false;
        };
        let mut cur = if target.is_absolute() {
            PathBuf::from("/")
        } else {
            match link.parent().map(std::fs::canonicalize) {
                Some(Ok(d)) => d,
                _ => return false,
            }
        };
        for c in target.components() {
            match c {
                Component::Normal(n) => cur.push(n),
                Component::ParentDir => {
                    cur.pop();
                }
                _ => {}
            }
        }
        cur.starts_with(&root) || cur.starts_with(&self.root)
    }

    fn pick_dir(&mut self, tc: &TestCase) -> PathBuf {
        let mut dirs = vec![self.root.clone(); 2];
        dirs.extend(self.known.iter().filter(|k| k.is_dir()).cloned());
        dirs.push(self.root.join("missing"));
        tc.draw(gs::sampled_from(dirs))
    }

    /// Forget temp files a faulted cleanup was entitled to leave: they have no owner left, and the
    /// spec's cleanup is one best-effort command. Returns the snapshot afterwards.
    fn settle_temps(&self, after: Snap, exempt: bool, what: &str) -> Snap {
        let t = temps(&after);
        if t.is_empty() {
            return after;
        }
        if !exempt {
            fail(format!(
                "{what} left a temp file behind: {:?}",
                t.iter()
                    .map(|p| p.strip_prefix(&self.root).unwrap().display().to_string())
                    .collect::<Vec<_>>()
            ));
        }
        for p in t {
            let _ = std::fs::remove_file(p);
        }
        snapshot(&self.root)
    }

    fn do_write(&mut self, tc: &TestCase, p: PathBuf, data: Vec<u8>, guard: Option<Option<u64>>) {
        let before = snapshot(&self.root);
        let cur = std::fs::metadata(&p)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(secs);
        let plan = self.with_space(truth::plan_write(&p), data.len());
        let res: Result<bool, FsError> = self.block(async {
            match guard {
                None => self
                    .fs
                    .write_bytes(&self.c(&p), data.clone())
                    .await
                    .map(|_| true),
                Some(e) => {
                    let exp = e.map(|s| SystemTime::UNIX_EPOCH + Duration::from_secs(s));
                    self.fs
                        .write_if_unchanged(&self.c(&p), data.clone(), exp)
                        .await
                }
            }
        });
        let report = self.faults.take();
        let after = self.settle_temps(snapshot(&self.root), report.cleanup_faulted, "a write");
        let mismatch = guard.map(|e| cur != e);
        if truth::resolve_leaf(&p).is_none() {
            // A symlink loop: the spec does not say what a write through one does. Only the temp
            // rule still applies (it was checked by `settle_temps` above).
            return;
        }
        let op = if guard.is_some() {
            "write_if_unchanged"
        } else {
            "write"
        };

        // What the tree looks like if the write was applied: only the resolved target changes.
        let target = truth::resolve_leaf(&p).unwrap_or(p.clone());
        let applied = {
            let mut e = before.clone();
            match e.get_mut(&target) {
                Some(Node::File { bytes, .. }) => *bytes = data.clone(),
                Some(_) => {}
                None => {
                    e.insert(
                        target.clone(),
                        Node::File {
                            mode: 0o644,
                            mtime: SystemTime::UNIX_EPOCH,
                            bytes: data.clone(),
                        },
                    );
                    if let (Some(par), Some(name)) = (target.parent(), target.file_name())
                        && let Some(Node::Dir { kids, .. }) = e.get_mut(par)
                    {
                        kids.insert(name.to_os_string());
                    }
                }
            }
            same_ignoring_mtimes(&after, &e, &[&target]).unwrap()
        };
        let check_applied = |what: &str| {
            if after != applied {
                fail(format!(
                    "{what}: the tree is not 'only the target changed to the new bytes':\n{}",
                    diff(&after, &applied, &self.root).join("\n")
                ));
            }
        };
        let check_unchanged = |what: &str| {
            if after != before {
                fail(format!(
                    "{what}: the tree changed:\n{}",
                    diff(&after, &before, &self.root).join("\n")
                ));
            }
        };
        tc.event(match (&res, &plan) {
            (Ok(true), _) => format!("{op}: applied"),
            (Ok(false), _) => format!("{op}: conflict"),
            (Err(_), _) if report.errored => format!("{op}: failed by an injected fault"),
            (Err(_), Plan::Fail(_)) => format!("{op}: refused as the spec requires"),
            (Err(_), _) => format!("{op}: failed"),
        });
        match &res {
            Ok(true) => {
                if let Plan::Fail(k) = &plan {
                    fail(format!(
                        "{op} reported success but the spec requires an error {k:?}; tree:\n{}",
                        diff(&after, &before, &self.root).join("\n")
                    ));
                }
                if mismatch == Some(true) {
                    fail(format!(
                        "lost update: {op} overwrote {p:?} though its mtime ({cur:?}) is not the expected {guard:?}"
                    ));
                }
                check_applied("an Ok write");
            }
            Ok(false) => {
                if mismatch != Some(true) {
                    fail(format!(
                        "spurious conflict: {op} refused {p:?} though mtime {cur:?} equals expected {guard:?}"
                    ));
                }
                check_unchanged("a conflict");
            }
            Err(e) => {
                let kind = self.kind(e, &p);
                if report.errored && matches!(kind, Kind::Transport | Kind::Timeout) {
                    // The spec's promise under a failed transport: unchanged or fully written, never
                    // partial, never a temp.
                    if mismatch == Some(true) || matches!(plan, Plan::Fail(_)) {
                        check_unchanged("a failed write");
                    } else if after != before {
                        check_applied("a write that failed after it ran");
                    }
                } else {
                    // The command itself failed (a fault may still have hit its cleanup).
                    match &plan {
                        Plan::Succeed
                            if kind == Kind::PermissionDenied
                                && std::fs::metadata(&target).is_ok_and(|m| m.is_file())
                                && !truth::can(&target, 4) =>
                        {
                            // The size check reads the temp back, and the temp has the file's mode.
                            known(
                                "N4",
                                format!(
                                    "{op} {p:?} refused: the file is writable but not readable by its owner: {e}"
                                ),
                            );
                        }
                        Plan::Succeed => fail(format!(
                            "{op} {p:?} must succeed but failed ({kind:?}): {e}"
                        )),
                        Plan::Fail(kinds) if !kinds.contains(&kind) => fail(format!(
                            "{op} {p:?} failed as {kind:?} ({e}), spec allows {kinds:?}"
                        )),
                        _ => {}
                    }
                    check_unchanged("a failed write");
                }
            }
        }
    }

    fn read_expect(&self, p: &Path, offset: u64, max: usize) -> Result<Vec<u8>, Vec<Kind>> {
        match std::fs::metadata(p) {
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                return Err(vec![Kind::NotFound, Kind::PermissionDenied]);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(vec![Kind::NotFound]),
            Err(_) => return Err(vec![Kind::NotFound, Kind::NotADirectory]),
            Ok(m) if m.is_dir() => return Err(vec![Kind::IsDirectory]),
            Ok(_) => {}
        }
        if !truth::can(p, 4) {
            return Err(vec![Kind::PermissionDenied]);
        }
        let data = std::fs::read(p).unwrap_or_default();
        let at = |i: u64| (i.min(data.len() as u64)) as usize;
        let mut out: Vec<u8> = vec![];
        loop {
            let want = (max - out.len()).min(96 * 1024);
            let start = at(offset.saturating_add(out.len() as u64));
            let n = want.min(data.len() - start);
            if let Some(cap) = self.cfg.cap {
                let resp = n.div_ceil(3) * 4;
                if resp > cap {
                    return Err(vec![Kind::TooLarge]);
                }
                if resp + 512 > cap {
                    // Inside the envelope's margin: either answer is the spec's.
                    return Err(vec![Kind::TooLarge, Kind::Io]);
                }
            }
            out.extend_from_slice(&data[start..start + n]);
            if n < want || out.len() >= max {
                return Ok(out);
            }
        }
    }
}

#[hegel::state_machine]
impl Machine {
    #[rule]
    fn write(&mut self, tc: TestCase) {
        let p = self.pick_path(&tc);
        let data = fixtures::content(&tc, self.cfg.cap, self.fill());
        tc.note(&format!(
            "write {:?} ({} bytes)",
            p.strip_prefix(&self.root).unwrap(),
            data.len()
        ));
        self.do_write(&tc, p, data, None);
    }

    #[rule]
    fn write_if_unchanged(&mut self, tc: TestCase) {
        let p = self.pick_path(&tc);
        let data = fixtures::content(&tc, self.cfg.cap, self.fill());
        let truth_now = std::fs::metadata(&p)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(secs);
        let remembered = self.stats.get(&p).copied();
        let expected = match tc.draw(gs::integers::<u8>().max_value(4)) {
            0 => None,
            1 => remembered.unwrap_or(None),
            2 => truth_now,
            3 => truth_now.map(|s| s + 1),
            _ => Some(secs(SystemTime::now())),
        };
        tc.note(&format!(
            "write_if_unchanged {:?} expected={expected:?} (mtime now {truth_now:?})",
            p.strip_prefix(&self.root).unwrap()
        ));
        self.do_write(&tc, p, data, Some(expected));
    }

    #[rule]
    fn stat(&mut self, tc: TestCase) {
        let p = self.pick_path(&tc);
        tc.note(&format!("stat {:?}", p.strip_prefix(&self.root).unwrap()));
        let before = snapshot(&self.root);
        let got = self.block(self.fs.stat(&self.c(&p)));
        let report = self.faults.take();
        let want = std::fs::metadata(&p).ok();
        match got {
            Err(e) => {
                if !report.errored {
                    fail(format!("stat {p:?} failed without a fault: {e}"));
                }
            }
            Ok(None) => {
                if let Some(m) = want
                    && !report.errored
                {
                    fail(format!(
                        "stat says {p:?} is absent but std::fs finds it: {m:?}"
                    ));
                }
            }
            Ok(Some(meta)) => {
                let Some(m) = want else {
                    fail(format!(
                        "stat says {p:?} exists ({meta:?}) but std::fs does not find it"
                    ));
                };
                let kind = if m.is_dir() {
                    FileKind::Dir
                } else if m.is_file() {
                    FileKind::File
                } else {
                    FileKind::Other
                };
                if meta.kind != kind {
                    fail(format!("stat kind {:?} != {kind:?} for {p:?}", meta.kind));
                }
                if meta.len != m.len() {
                    fail(format!(
                        "stat len {} != {} for {p:?} (following symlinks)",
                        meta.len,
                        m.len()
                    ));
                }
                let ts = m.modified().map(secs).unwrap();
                if meta.mtime.map(secs) != Some(ts) {
                    fail(format!(
                        "stat mtime {:?} != {ts} for {p:?} (following symlinks)",
                        meta.mtime.map(secs)
                    ));
                }
                if meta.writable != truth::can(&p, 2) {
                    fail(format!(
                        "stat writable {} != {} for {p:?}",
                        meta.writable,
                        truth::can(&p, 2)
                    ));
                }
                self.stats.insert(p.clone(), meta.mtime.map(secs));
            }
        }
        read_only_ops_unchanged(self, &before, "stat");
    }

    #[rule]
    fn read(&mut self, tc: TestCase) {
        let p = self.pick_path(&tc);
        let size = std::fs::metadata(&p).map_or(0, |m| m.len());
        let offset = tc.draw(gs::sampled_from(vec![
            0,
            0,
            1,
            size / 2,
            size.saturating_sub(1),
            size,
            size + 1,
            size * 2 + 7,
            98304,
            1 << 40,
        ]));
        let s = size as usize;
        let max = tc.draw(gs::sampled_from(vec![
            0usize,
            1,
            7,
            s,
            s + 1,
            48 * 1024,
            98303,
            98304,
            98305,
            2 * 98304,
            1 << 20,
            usize::MAX,
        ]));
        tc.note(&format!(
            "read {:?} offset={offset} max={max} (size {size})",
            p.strip_prefix(&self.root).unwrap()
        ));
        let before = snapshot(&self.root);
        let got = self.block(self.fs.read_bytes(&self.c(&p), offset, max));
        let report = self.faults.take();
        let want = self.read_expect(&p, offset, max);
        tc.event(format!("read: {}", if got.is_ok() { "ok" } else { "err" }));
        match (got, want) {
            (Ok(b), Ok(w)) => {
                if b != w {
                    fail(format!(
                        "read bytes differ: got {} bytes, std::fs range has {} (first difference at {:?})",
                        b.len(),
                        w.len(),
                        b.iter().zip(&w).position(|(x, y)| x != y)
                    ));
                }
            }
            (Err(e), _) if report.errored => {
                let k = self.kind(&e, &p);
                if !matches!(k, Kind::Transport | Kind::Timeout) {
                    fail(format!("injected fault surfaced as {k:?}: {e}"));
                }
            }
            (Ok(b), Err(kinds)) => {
                if !(kinds.contains(&Kind::TooLarge) && kinds.contains(&Kind::Io)) {
                    fail(format!(
                        "read of {p:?} returned {} bytes but the spec requires an error {kinds:?}",
                        b.len()
                    ));
                }
                // inside the margin, a success is fine; it must still be right
                let w = std::fs::read(&p).unwrap_or_default();
                let lo = (offset as usize).min(w.len());
                let hi = lo.saturating_add(max).min(w.len());
                if b != w[lo..hi] {
                    fail(format!("read bytes differ inside the cap margin for {p:?}"));
                }
            }
            (Err(e), Ok(w)) => fail(format!(
                "read {p:?} must return {} bytes but failed: {e}",
                w.len()
            )),
            (Err(e), Err(kinds)) => {
                let k = self.kind(&e, &p);
                if !kinds.contains(&k) {
                    fail(format!(
                        "read {p:?} failed as {k:?} ({e}); spec allows {kinds:?}"
                    ));
                }
            }
        }
        read_only_ops_unchanged(self, &before, "read");
    }

    #[rule]
    fn list(&mut self, tc: TestCase) {
        let p = if tc.draw(gs::booleans()) {
            self.pick_dir(&tc)
        } else {
            self.pick_path(&tc)
        };
        let hidden = tc.draw(gs::booleans());
        let cap = tc.draw(gs::sampled_from(vec![0usize, 1, 2, 5, 10_000]));
        tc.note(&format!(
            "list {:?} hidden={hidden} cap={cap}",
            p.strip_prefix(&self.root).unwrap()
        ));
        let before = snapshot(&self.root);
        let got = self.block(self.fs.list_dir(&self.c(&p), cap, hidden));
        let report = self.faults.take();
        let meta = std::fs::metadata(&p);
        let mut want: Vec<(String, u8, u64)> = vec![];
        let mut readable = true;
        match &meta {
            Ok(m) if m.is_dir() => {
                if truth::can(&p, 5) {
                    for e in std::fs::read_dir(&p).unwrap().flatten() {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if !hidden && name.starts_with('.') {
                            continue;
                        }
                        let Ok(follow) = std::fs::metadata(e.path()) else {
                            continue;
                        };
                        let own = std::fs::symlink_metadata(e.path()).unwrap();
                        let k = if follow.is_dir() {
                            1
                        } else if follow.is_file() {
                            0
                        } else {
                            2
                        };
                        want.push((name, k, own.len()));
                    }
                } else {
                    readable = false;
                }
            }
            _ => {}
        }
        match got {
            Err(e) => {
                let k = self.kind(&e, &p);
                if report.errored {
                    return;
                }
                let ok = match &meta {
                    Err(_) => {
                        matches!(
                            k,
                            Kind::NotFound | Kind::PermissionDenied | Kind::NotADirectory
                        ) || std::fs::symlink_metadata(&p).is_ok()
                    }
                    Ok(_) => !readable && matches!(k, Kind::PermissionDenied | Kind::Io),
                };
                if !ok {
                    fail(format!("list {p:?} failed as {k:?}: {e}"));
                }
            }
            Ok(entries) => {
                let dangling = meta.is_err()
                    && std::fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink());
                if meta.is_err() && !dangling && !report.errored {
                    fail(format!(
                        "list of the missing {p:?} succeeded with {} entries",
                        entries.len()
                    ));
                }
                if matches!(&meta, Ok(m) if !m.is_dir()) && !entries.is_empty() {
                    fail(format!("list of the file {p:?} returned entries"));
                }
                if readable {
                    let mut got: Vec<(String, u8, u64)> = entries
                        .iter()
                        .map(|e| {
                            (
                                e.name.clone(),
                                match e.kind {
                                    FileKind::File => 0,
                                    FileKind::Dir => 1,
                                    FileKind::Other => 2,
                                },
                                e.len,
                            )
                        })
                        .collect();
                    got.sort();
                    want.sort();
                    if want.len() <= cap {
                        if got != want {
                            fail(format!("list {p:?}:\n got  {got:?}\n want {want:?}"));
                        }
                    } else {
                        if got.len() != cap {
                            fail(format!(
                                "list {p:?} cap={cap} returned {} of {} kept entries",
                                got.len(),
                                want.len()
                            ));
                        }
                        if let Some(g) = got.iter().find(|g| !want.contains(g)) {
                            fail(format!("list {p:?} returned {g:?}, not in {want:?}"));
                        }
                    }
                }
            }
        }
        read_only_ops_unchanged(self, &before, "list");
    }

    #[rule]
    fn search(&mut self, tc: TestCase) {
        let root = if tc.draw(gs::integers::<u8>().max_value(7)) < 4 {
            self.root.clone()
        } else {
            self.pick_path(&tc)
        };
        let (pattern, literal, ignore_case) = fixtures::query(&tc);
        let glob = match tc.draw(gs::integers::<u8>().max_value(5)) {
            0 => Some(("*.txt".to_string(), false)),
            1 => Some(("*.txt".to_string(), true)),
            _ => None,
        };
        let limit = tc.draw(gs::sampled_from(vec![0usize, 1, 2, 3, 100]));
        let before_n = tc.draw(gs::sampled_from(vec![0usize, 0, 1, 2, 150]));
        let after_n = tc.draw(gs::sampled_from(vec![0usize, 0, 1, 2, 150]));
        let spec = Spec {
            pattern: pattern.clone(),
            literal,
            ignore_case,
            glob: glob.clone(),
            limit,
            before: before_n,
            after: after_n,
        };
        tc.note(&format!(
            "search {pattern:?} in {:?} literal={literal} ic={ignore_case} glob={glob:?} limit={limit} -B{before_n} -A{after_n}",
            root.strip_prefix(&self.root).unwrap_or(&root)
        ));
        let before = snapshot(&self.root);
        let q = SearchQuery {
            pattern,
            literal,
            ignore_case,
            glob,
            root: self.c(&root),
            limit,
            before: before_n,
            after: after_n,
        };
        let got = self.block(self.fs.search(&q));
        let report = self.faults.take();
        let map = |p: &Path| self.c(p).to_string_lossy().into_owned();
        let want = search_truth::expect(&root, &map, &spec, self.engine);
        tc.event(format!(
            "search: {} {}",
            if want.exact {
                "exact"
            } else {
                "soundness-only"
            },
            if got.is_ok() { "ok" } else { "err" }
        ));
        if let Ok(o) = &got {
            tc.event(format!(
                "search: {}",
                if o.hits.is_empty() { "no hits" } else { "hits" }
            ));
            tc.event(format!(
                "search: {}",
                if o.truncated { "truncated" } else { "complete" }
            ));
        }
        match got {
            Err(e) => {
                let k = self.kind(&e, &root);
                let tolerated = report.errored || (self.cfg.cap.is_some() && k == Kind::TooLarge);
                if !tolerated {
                    if want.root_missing {
                        let expect = if want.root_below_file {
                            Kind::NotADirectory
                        } else {
                            Kind::NotFound
                        };
                        if k != expect {
                            fail(format!(
                                "search of the missing root {root:?} must fail as {expect:?}, got {k:?}: {e}"
                            ));
                        }
                    } else if want.unreadable {
                        fail(format!(
                            "search lost the matches it could read because a path was unreadable ({k:?}): {e}"
                        ));
                    } else {
                        fail(format!("search {:?} failed ({k:?}): {e}", spec.pattern));
                    }
                }
            }
            Ok(out) => {
                if want.root_missing && !report.errored {
                    fail(format!(
                        "search of the missing root {root:?} succeeded with {} hits",
                        out.hits.len()
                    ));
                }
                let got_hits: Vec<Th> = out
                    .hits
                    .iter()
                    .map(|h| Th {
                        path: h.path.to_string_lossy().into_owned(),
                        line: h.line,
                        text: h.text.clone(),
                        is_match: h.is_match,
                    })
                    .collect();
                if want.unreadable && !want.root_missing && out.first_error.is_none() {
                    fail("search over an unreadable path reported no note");
                }
                let key = |h: &Th| {
                    (
                        h.path
                            .split('/')
                            .map(|c| c.as_bytes().to_vec())
                            .collect::<Vec<_>>(),
                        h.line,
                    )
                };
                if got_hits.windows(2).any(|w| key(&w[0]) > key(&w[1])) {
                    fail(format!(
                        "search hits are not sorted by (path components, line): {:?}",
                        got_hits
                            .iter()
                            .map(|h| (&h.path, h.line))
                            .collect::<Vec<_>>()
                    ));
                }
                if got_hits.iter().filter(|h| h.is_match).count() > limit {
                    fail(format!("search returned more than limit={limit} matches"));
                }
                if out.truncated && got_hits.last().is_some_and(|h| !h.is_match) {
                    fail(
                        "search result ends on a context line (trailing context must be dropped when truncated)",
                    );
                }
                if want.exact {
                    if got_hits != want.hits {
                        let first = got_hits
                            .iter()
                            .zip(&want.hits)
                            .position(|(a, b)| a != b)
                            .unwrap_or(got_hits.len().min(want.hits.len()));
                        fail(format!(
                            "search differs from the brute-force search at hit {first}:\n got  ({}) {:?}\n want ({}) {:?}",
                            got_hits.len(),
                            got_hits.get(first),
                            want.hits.len(),
                            want.hits.get(first)
                        ));
                    }
                    if out.truncated != want.truncated {
                        fail(format!(
                            "search truncated={} but expected {}",
                            out.truncated, want.truncated
                        ));
                    }
                } else if !want.root_missing {
                    for h in &got_hits {
                        if !want.hits.contains(h) {
                            fail(format!(
                                "search returned a hit that is not in the tree: {h:?}"
                            ));
                        }
                    }
                }
            }
        }
        read_only_ops_unchanged(self, &before, "search");
    }

    #[rule]
    fn mkdir(&mut self, tc: TestCase) {
        let p = self.pick_path(&tc);
        tc.note(&format!(
            "mkdir -p {:?}",
            p.strip_prefix(&self.root).unwrap()
        ));
        let before = snapshot(&self.root);
        let plan = truth::plan_mkdir(&p);
        let res = self.block(self.fs.create_dir_all(&self.c(&p)));
        let report = self.faults.take();
        let after = snapshot(&self.root);
        let resolved = truth::canon(&p).unwrap_or_else(|_| p.clone());
        let new_ok = |after: &Snap| {
            // Only directories on the way to `p` may appear.
            after.iter().all(|(k, n)| match before.get(k) {
                Some(Node::Dir { mode, kids }) => matches!(n, Node::Dir { mode: m2, kids: k2 } if m2 == mode && kids.is_subset(k2)),
                Some(b) => b == n,
                None => matches!(n, Node::Dir { .. }) && (p.starts_with(k) || resolved.starts_with(k)),
            }) && before.keys().all(|k| after.contains_key(k))
        };
        match res {
            Ok(()) => {
                if let Plan::Fail(k) = plan {
                    fail(format!(
                        "mkdir -p {p:?} reported success but the spec requires an error {k:?}"
                    ));
                }
                if !std::fs::metadata(&p).is_ok_and(|m| m.is_dir()) {
                    fail(format!(
                        "mkdir -p {p:?} reported success but there is no directory"
                    ));
                }
                if !new_ok(&after) {
                    fail(format!(
                        "mkdir changed more than the path:\n{}",
                        diff(&after, &before, &self.root).join("\n")
                    ));
                }
            }
            Err(e) => {
                let k = self.kind(&e, &p);
                if !report.errored {
                    match plan {
                        Plan::Succeed => fail(format!(
                            "mkdir -p {p:?} must succeed but failed ({k:?}): {e}"
                        )),
                        Plan::Fail(ks) if !ks.contains(&k) => fail(format!(
                            "mkdir -p {p:?} failed as {k:?}, spec allows {ks:?}"
                        )),
                        _ => {}
                    }
                    if after != before {
                        fail(format!(
                            "a failed mkdir changed the tree:\n{}",
                            diff(&after, &before, &self.root).join("\n")
                        ));
                    }
                } else if !new_ok(&after) {
                    fail("a faulted mkdir changed more than the path");
                }
            }
        }
    }

    /// Two writers to one path: one complete content wins; a reader sees only whole contents.
    #[rule]
    fn race_write(&mut self, tc: TestCase) {
        tc.assume(self.cfg.fault_pct == 0 && self.cfg.tiny_kib.is_none());
        let p = self.pick_path(&tc);
        tc.assume(matches!(truth::plan_write(&p), Plan::Succeed));
        // (N4: a file its owner cannot read cannot be replaced, so a race over one proves nothing.)
        tc.assume(
            std::fs::metadata(&p).is_ok_and(|m| m.is_dir() || truth::can(&p, 4))
                || std::fs::metadata(&p).is_err(),
        );
        tc.assume(
            std::fs::symlink_metadata(&p).is_ok_and(|m| m.is_file())
                || std::fs::symlink_metadata(&p).is_err(),
        );
        let a = fixtures::content(&tc, self.cfg.cap, self.fill());
        let b = fixtures::content(&tc, self.cfg.cap, self.fill());
        let mode = tc.draw(gs::integers::<u8>().max_value(2));
        tc.note(&format!(
            "race_write {:?} a={}B b={}B second={}",
            p.strip_prefix(&self.root).unwrap(),
            a.len(),
            b.len(),
            ["none (observer only)", "second ShellFs", "std::fs::write"][mode as usize]
        ));
        let old = std::fs::read(&p).ok();
        let stop = Arc::new(AtomicBool::new(false));
        let seen: Arc<Mutex<Vec<Option<Vec<u8>>>>> = Arc::default();
        let observer = (mode != 2).then(|| {
            let (stop, seen, p) = (stop.clone(), seen.clone(), p.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let r = std::fs::read(&p).ok();
                    let mut s = seen.lock().unwrap();
                    if !s.contains(&r) {
                        s.push(r);
                    }
                    drop(s);
                    std::thread::yield_now();
                }
            })
        });
        let (r1, r2) = self.block(async {
            let first = self.fs.write_bytes(&p, a.clone());
            let second = async {
                match mode {
                    1 => self.fs2.write_bytes(&p, b.clone()).await,
                    2 => {
                        let (pp, bb) = (p.clone(), b.clone());
                        tokio::task::spawn_blocking(move || std::fs::write(pp, bb))
                            .await
                            .unwrap()
                            .map_err(|e| FsError::Backend(e.to_string()))
                    }
                    _ => Ok(()),
                }
            };
            tokio::join!(first, second)
        });
        stop.store(true, Ordering::Relaxed);
        if let Some(o) = observer {
            o.join().unwrap();
        }
        let after = snapshot(&self.root);
        let t = temps(&after);
        if !t.is_empty() {
            fail(format!("racing writers left temp files: {t:?}"));
        }
        if let Err(e) = r1 {
            fail(format!("racing write failed: {e}"));
        }
        if let Err(e) = r2 {
            fail(format!("second racing write failed: {e}"));
        }
        let fin = std::fs::read(&p).unwrap_or_default();
        let allowed: Vec<&Vec<u8>> = [Some(&a), (mode > 0).then_some(&b)]
            .into_iter()
            .flatten()
            .collect();
        if !allowed.contains(&&fin) {
            fail(format!(
                "after two writers the file is neither complete content: {} bytes (a={}, b={})",
                fin.len(),
                a.len(),
                b.len()
            ));
        }
        for s in seen.lock().unwrap().iter() {
            let ok = match s {
                None => old.is_none(),
                Some(x) => Some(x) == old.as_ref() || allowed.contains(&x),
            };
            if !ok {
                fail(format!(
                    "a reader saw a partial/mixed file during a write: {} bytes (old={:?}, a={}, b={})",
                    s.as_ref().map_or(0, Vec::len),
                    old.as_ref().map(Vec::len),
                    a.len(),
                    b.len()
                ));
            }
        }
    }

    // ---- environment: changes made directly on disk, behind ShellFs's back ----

    #[rule]
    fn env_write(&mut self, tc: TestCase) {
        let p = self.pick_path(&tc);
        let data = fixtures::content(&tc, self.cfg.cap, self.fill());
        // `keep` re-sets the previous mtime: a same-second edit the guard cannot see (a documented
        // limit); the others move the mtime to a different second.
        let how = tc.draw(gs::integers::<u8>().max_value(3));
        tc.note(&format!(
            "env: write {:?} {}B mtime-mode {how}",
            p.strip_prefix(&self.root).unwrap(),
            data.len()
        ));
        let prev = std::fs::metadata(&p).ok().and_then(|m| m.modified().ok());
        if std::fs::write(&p, &data).is_ok()
            && let Ok(f) = std::fs::File::options().write(true).open(&p)
        {
            let t = match how {
                0 => prev.unwrap_or_else(SystemTime::now),
                1 => SystemTime::now() + Duration::from_secs(3),
                2 => SystemTime::now() - Duration::from_secs(3600),
                _ => SystemTime::now(),
            };
            let _ = f.set_modified(t);
        }
    }

    #[rule]
    fn env_mkdir(&mut self, tc: TestCase) {
        let p = self.pick_path(&tc);
        tc.note(&format!(
            "env: mkdir {:?}",
            p.strip_prefix(&self.root).unwrap()
        ));
        let _ = std::fs::create_dir_all(&p);
    }

    #[rule]
    fn env_symlink(&mut self, tc: TestCase) {
        tc.assume(self.cfg.tiny_kib.is_none());
        let link = self.pick_path(&tc);
        let target = match tc.draw(gs::integers::<u8>().max_value(4)) {
            0 => PathBuf::from("/dev/null"),
            1 => self.root.join("gone"),
            2 => PathBuf::from(fixtures::name(&tc)).with_extension("lnk"),
            _ => self.pick_path(&tc),
        };
        // A link must never lead out of the test root: `list` follows it, and the oracle and the
        // client would then read a live, shared directory (`/tmp`) at different moments. A link to a
        // file outside (`/dev/null`) is harmless; a dangling one inside is fine.
        if target != Path::new("/dev/null") && !self.stays_inside(&link, &target) {
            return;
        }
        tc.note(&format!(
            "env: symlink {:?} -> {:?}",
            link.strip_prefix(&self.root).unwrap(),
            target.strip_prefix(&self.root).unwrap_or(&target)
        ));
        if target != link {
            let _ = std::os::unix::fs::symlink(&target, &link);
        }
    }

    #[rule]
    fn env_chmod(&mut self, tc: TestCase) {
        // Inside the namespace the caller is root-mapped, so mode bits do not bind there.
        tc.assume(self.cfg.tiny_kib.is_none());
        let p = self.pick_path(&tc);
        let mode = tc.draw(gs::sampled_from(vec![
            0o000u32, 0o444, 0o555, 0o600, 0o644, 0o700, 0o755, 0o111, 0o333,
        ]));
        tc.note(&format!(
            "env: chmod {mode:o} {:?}",
            p.strip_prefix(&self.root).unwrap()
        ));
        if std::fs::symlink_metadata(&p).is_ok_and(|m| !m.file_type().is_symlink()) {
            let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode));
        }
    }

    #[rule]
    fn env_remove(&mut self, tc: TestCase) {
        let p = self.pick_path(&tc);
        tc.note(&format!(
            "env: remove {:?}",
            p.strip_prefix(&self.root).unwrap()
        ));
        let _ = snapshot(&self.root); // unlock, so a locked directory can be emptied
        if std::fs::symlink_metadata(&p).is_ok_and(|m| m.is_dir()) {
            let _ = std::fs::remove_dir_all(&p);
        } else {
            let _ = std::fs::remove_file(&p);
        }
    }

    /// A file whose name is not UTF-8: it cannot be addressed over JSON, so it only appears in
    /// listings and search results.
    #[rule]
    fn env_non_utf8_name(&mut self, tc: TestCase) {
        use std::os::unix::ffi::OsStrExt;
        let dir = self.pick_dir(&tc);
        tc.note(&format!(
            "env: non-UTF-8 file in {:?}",
            dir.strip_prefix(&self.root).unwrap_or(&dir)
        ));
        let name = std::ffi::OsStr::from_bytes(b"bad\xff\xfename");
        let _ = std::fs::write(dir.join(name), b"needle in a bad name\n");
    }
}
