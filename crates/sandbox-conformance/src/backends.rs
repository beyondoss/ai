//! The backends: an exec endpoint's URL, plus how to start and stop it.
//!
//! A [`Started`] backend is a URL, an optional bearer, a directory scenarios may use, and a guard
//! whose drop stops it. Adding a live backend is a function that returns one of these from a URL and
//! a bearer in the environment; nothing else in the harness knows what a backend is made of.
use std::any::Any;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use beyond_test_support::exec_mock::{ExecMock, MockOptions};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt as _, BufReader};
use tokio::process::Command;

const BEARER: &str = "conformance-secret";

/// How long a started endpoint has to say it is listening.
const READY_WITHIN: Duration = Duration::from_secs(60);

/// Every backend name, and the groups that expand to several.
pub const GROUPS: &[(&str, &[&str])] = &[(
    "local",
    &[
        "local-plain",
        "local-bearer",
        "local-grep",
        "local-busybox-grep",
    ],
)];
pub const SINGLES: &[&str] = &["alpine", "debian-runas", "readonly", "tinydisk"];

/// Expand group names and `*` globs to backend names, in a stable order, without duplicates.
pub fn expand(requested: &[String]) -> Result<Vec<String>, String> {
    // `e2b` costs money and needs the internet: only ever by its exact name, never in a group or a glob.
    let (live, requested): (Vec<&String>, Vec<&String>) = requested
        .iter()
        .partition(|w| *w == "e2b" || *w == "beyond");
    if live.iter().any(|w| *w == "e2b") && crate::e2b::key().is_none() {
        return Err("backend e2b needs E2B_API_KEY in the environment".into());
    }
    if live.iter().any(|w| *w == "beyond") {
        crate::beyond::configured()?;
    }
    let requested: Vec<String> = requested.into_iter().cloned().collect();
    let requested = &requested[..];
    let all: Vec<&str> = GROUPS
        .iter()
        .flat_map(|(_, m)| m.iter().copied())
        .chain(SINGLES.iter().copied())
        .collect();
    let mut out: Vec<String> = Vec::new();
    for want in requested {
        let found: Vec<&str> = if let Some((_, members)) = GROUPS.iter().find(|(g, _)| g == want) {
            members.to_vec()
        } else {
            all.iter()
                .copied()
                .filter(|n| crate::run::glob_match(want, n))
                .collect()
        };
        if found.is_empty() {
            return Err(format!(
                "unknown backend {want:?}; known: {}",
                all.join(", ")
            ));
        }
        for n in found {
            if !out.iter().any(|o| o == n) {
                out.push(n.to_string());
            }
        }
    }
    for w in ["e2b", "beyond"] {
        if live.iter().any(|l| *l == w) {
            out.push(w.into());
        }
    }
    Ok(out)
}

pub type Lease = std::sync::Arc<
    dyn Fn() -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<std::time::Duration, String>> + Send>,
        > + Send
        + Sync,
>;

pub struct Started {
    pub name: String,
    pub url: String,
    pub bearer: Option<String>,
    pub work: String,
    /// `CONFORMANCE_CAPS`: probe results the driver is told to believe.
    pub caps: Option<Value>,
    /// The suites this backend runs; `None` is all of them.
    pub suites: Option<&'static [&'static str]>,
    pub vars: BTreeMap<String, String>,
    /// Called before each cell: renews the backend's lifetime and returns how long the cell may take.
    /// `None` for a backend that cannot expire.
    pub lease: Option<Lease>,
    /// What a `host` step may do to this backend's machine; `None` when the harness cannot.
    pub host: Option<std::sync::Arc<dyn crate::run::Host>>,
    pub(crate) _guard: Box<dyn Any + Send + Sync>,
}

/// Where the binaries come from.
#[derive(Clone, Debug, Default)]
pub struct Binaries {
    /// execd for this host (the `local` backends).
    pub execd: Option<PathBuf>,
    /// A static (musl) execd, mounted into the containers.
    pub execd_static: Option<PathBuf>,
}

impl Binaries {
    fn host(&self) -> Result<PathBuf, String> {
        find(
            &self.execd,
            "EXECD_BIN",
            &["execd"],
            "build it: cargo build -p beyond-execd",
        )
    }

    fn container(&self) -> Result<PathBuf, String> {
        find(
            &self.execd_static,
            "EXECD_STATIC_BIN",
            &[
                "../x86_64-unknown-linux-musl/debug/execd",
                "../x86_64-unknown-linux-musl/release/execd",
                "../aarch64-unknown-linux-musl/debug/execd",
                "../aarch64-unknown-linux-musl/release/execd",
            ],
            "build it: mise run build:execd-static",
        )
    }
}

fn find(given: &Option<PathBuf>, env: &str, rel: &[&str], hint: &str) -> Result<PathBuf, String> {
    if let Some(p) = given
        .clone()
        .or_else(|| std::env::var_os(env).map(PathBuf::from))
    {
        return p
            .canonicalize()
            .map_err(|e| format!("{}: {e}", p.display()));
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let dir = exe.parent().unwrap_or(Path::new("."));
    rel.iter()
        .map(|r| dir.join(r))
        .find_map(|p| p.canonicalize().ok().filter(|p| p.is_file()))
        .ok_or_else(|| format!("no execd binary (${env}, or next to sandbox-matrix); {hint}"))
}

pub async fn start(name: &str, bins: &Binaries) -> Result<Started, String> {
    match name {
        "local-plain" => local(name, bins, false, None).await,
        "local-bearer" => local(name, bins, true, None).await,
        "local-grep" => local(name, bins, false, Some(json!({"search_engine": "grep"}))).await,
        "local-busybox-grep" => {
            local(
                name,
                bins,
                false,
                Some(json!({"search_engine": "busybox-grep", "find_printf": false})),
            )
            .await
        }
        // execd's own library behind a loopback listener, in this process: for the harness's own tests,
        // which need no binary. Not part of any group.
        "embedded" => embedded(name).await,
        "e2b" => crate::e2b::start(name).await,
        "beyond" => crate::beyond::start(name).await,
        "alpine" => {
            docker(
                name,
                bins,
                "alpine:3.20",
                Container::default().adduser(false),
            )
            .await
        }
        "debian-runas" => docker(name, bins, "debian:12-slim", Container::default()).await,
        "readonly" => {
            let ro = tempfile::tempdir().map_err(|e| e.to_string())?;
            std::fs::write(ro.path().join("existing"), b"old\n").map_err(|e| e.to_string())?;
            {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::set_permissions(ro.path(), std::fs::Permissions::from_mode(0o755))
                    .map_err(|e| e.to_string())?;
            }
            let mut c = Container::default();
            c.args.push("-v".into());
            c.args.push(format!("{}:/ro:ro", ro.path().display()));
            // A directory the sandbox user may not write, on a writable filesystem: the other way
            // to be refused.
            c.setup = "mkdir -p /locked && echo old > /locked/existing && chmod 555 /locked".into();
            c.vars.insert("RO".into(), "/ro".into());
            c.vars.insert("LOCKED".into(), "/locked".into());
            c.keep = Some(Box::new(ro));
            docker(name, bins, "debian:12-slim", c).await
        }
        "tinydisk" => {
            let mut c = Container::default();
            c.args.push("--tmpfs".into());
            c.args.push("/tiny:size=1m,mode=1777".into());
            c.vars.insert("TINY".into(), "/tiny".into());
            docker(name, bins, "debian:12-slim", c).await
        }
        other => Err(format!("unknown backend {other:?}")),
    }
}

// ---- local execd -------------------------------------------------------------------------------

struct Local {
    _host: std::sync::Arc<LocalHost>,
    _dirs: (tempfile::TempDir, tempfile::TempDir, tempfile::TempDir),
}

/// The local execd, as a process the harness may kill and start again (the `host` step).
struct LocalHost {
    execd: PathBuf,
    root: PathBuf,
    jobs: PathBuf,
    /// `EXECD_STATE_DIR`: the jobs ledger, which a restarted execd reads to find its jobs.
    state: PathBuf,
    bearer: bool,
    /// `127.0.0.1:PORT` the first execd bound; a restarted one binds the same, so the URL holds.
    addr: String,
    proc: tokio::sync::Mutex<(tokio::process::Child, Option<crate::reaper::Leader>)>,
}

impl LocalHost {
    /// Start execd on `listen` (`127.0.0.1:0` the first time): its job directory is private to this
    /// backend, so its terminal keepers can be told from every other backend's.
    async fn spawn(
        execd: &Path,
        root: &Path,
        (jobs, state): (&Path, &Path),
        bearer: bool,
        listen: &str,
    ) -> Result<(tokio::process::Child, Option<crate::reaper::Leader>, String), String> {
        let mut c = Command::new(execd);
        c.env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            // port-0: execd binds it, keeps it, and announces the port it got (read by `listening`).
            .env("EXECD_LISTEN", listen)
            .env("EXECD_ROOT", root)
            .env("EXECD_JOBS_DIR", jobs)
            .env("EXECD_STATE_DIR", state)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .process_group(0);
        if bearer {
            c.env("EXECD_BEARER", BEARER);
        }
        let mut child = c.spawn().map_err(|e| format!("start execd: {e}"))?;
        let leader = crate::reaper::Leader::new(&child);
        let addr = listening(child.stderr.take().expect("piped")).await?;
        Ok((child, leader, addr))
    }

    /// The terminal keepers (`execd --execd-run-as-helper keep <spool> ..`) of spools under this
    /// backend's job directory.
    fn keepers(&self) -> Vec<String> {
        let needle = self.jobs.to_string_lossy().into_owned();
        let Ok(dir) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        dir.flatten()
            .filter_map(|e| {
                let raw = std::fs::read(e.path().join("cmdline")).ok()?;
                let text = String::from_utf8_lossy(&raw).replace('\0', " ");
                (text.contains("--execd-run-as-helper keep ") && text.contains(&needle))
                    .then(|| e.file_name().to_string_lossy().into_owned())
                    .filter(|p| p.bytes().all(|b| b.is_ascii_digit()))
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl crate::run::Host for LocalHost {
    fn actions(&self) -> &'static [&'static str] {
        &["restart_execd", "kill_keepers"]
    }

    async fn act(&self, action: &str) -> Result<Value, String> {
        match action {
            "restart_execd" => {
                let mut p = self.proc.lock().await;
                // SIGHUP is execd's restart: it exits and leaves its jobs running (a stop, SIGTERM,
                // ends them). A keeper is in a group of its own, so it outlives execd, as it would an
                // upgrade or a restore. The old group's leader entry goes with it (its drop kills a
                // group that now holds nothing).
                let pid = p.0.id().ok_or("execd already exited")?;
                let sent = Command::new("kill")
                    .args(["-HUP", &pid.to_string()])
                    .status()
                    .await
                    .map_err(|e| format!("kill -HUP: {e}"))?;
                if !sent.success() {
                    return Err(format!("kill -HUP {pid} failed"));
                }
                tokio::time::timeout(READY_WITHIN, p.0.wait())
                    .await
                    .map_err(|_| "execd did not exit on SIGHUP".to_string())?
                    .map_err(|e| format!("wait for execd: {e}"))?;
                drop(p.1.take());
                let (child, leader, addr) = Self::spawn(
                    &self.execd,
                    &self.root,
                    (&self.jobs, &self.state),
                    self.bearer,
                    &self.addr,
                )
                .await?;
                if addr != self.addr {
                    return Err(format!("execd came back on {addr}, not {}", self.addr));
                }
                *p = (child, leader);
                Ok(json!({}))
            }
            "kill_keepers" => {
                let pids = self.keepers();
                for pid in &pids {
                    let _ = Command::new("kill").args(["-KILL", pid]).status().await;
                }
                Ok(json!({ "killed": pids.len() }))
            }
            other => Err(format!("no host action {other:?}")),
        }
    }
}

async fn local(
    name: &str,
    bins: &Binaries,
    bearer: bool,
    caps: Option<Value>,
) -> Result<Started, String> {
    let execd = bins.host()?;
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let work = dir.path().canonicalize().map_err(|e| e.to_string())?;
    let jobs_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let jobs = jobs_dir.path().canonicalize().map_err(|e| e.to_string())?;
    let state_dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let state = state_dir.path().canonicalize().map_err(|e| e.to_string())?;
    let (child, leader, addr) =
        LocalHost::spawn(&execd, &work, (&jobs, &state), bearer, "127.0.0.1:0").await?;
    let host = std::sync::Arc::new(LocalHost {
        execd,
        root: work.clone(),
        jobs,
        state,
        bearer,
        addr: addr.clone(),
        proc: tokio::sync::Mutex::new((child, leader)),
    });
    Ok(Started {
        name: name.into(),
        url: format!("http://{addr}/exec"),
        bearer: bearer.then(|| BEARER.to_string()),
        work: work.to_string_lossy().into_owned(),
        caps,
        suites: None,
        vars: BTreeMap::new(),
        lease: None,
        host: Some(host.clone()),
        _guard: Box::new(Local {
            _host: host,
            _dirs: (dir, jobs_dir, state_dir),
        }),
    })
}

/// The address from execd's `listening listen=ADDR` line. The pipe is then drained so execd never
/// blocks on a full stderr.
async fn listening(stderr: tokio::process::ChildStderr) -> Result<String, String> {
    let mut lines = BufReader::new(stderr).lines();
    let found = tokio::time::timeout(READY_WITHIN, async {
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(rest) = line.split(" listen=").nth(1) {
                return Some(rest.split_whitespace().next().unwrap_or("").to_string());
            }
        }
        None
    })
    .await;
    match found {
        Ok(Some(addr)) => {
            tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
            Ok(addr)
        }
        _ => Err("execd did not report a listening address".into()),
    }
}

// ---- execd in this process ---------------------------------------------------------------------

async fn embedded(name: &str) -> Result<Started, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let work = dir.path().canonicalize().map_err(|e| e.to_string())?;
    let mock = ExecMock::start_with(&work, MockOptions::default()).await;
    Ok(Started {
        name: name.into(),
        url: mock.url.clone(),
        bearer: None,
        work: work.to_string_lossy().into_owned(),
        caps: None,
        suites: None,
        vars: BTreeMap::new(),
        lease: None,
        host: None,
        _guard: Box::new((mock, dir)),
    })
}

// ---- execd in a container ----------------------------------------------------------------------

struct Container {
    /// Extra `docker run` arguments.
    args: Vec<String>,
    /// A shell fragment run as root before execd starts.
    setup: String,
    adduser: bool,
    vars: BTreeMap<String, String>,
    keep: Option<Box<dyn Any + Send + Sync>>,
}

impl Default for Container {
    fn default() -> Self {
        Self {
            args: Vec::new(),
            setup: String::new(),
            adduser: true,
            vars: BTreeMap::new(),
            keep: None,
        }
    }
}

impl Container {
    fn adduser(mut self, on: bool) -> Self {
        self.adduser = on;
        self
    }
}

/// Removes the container when the backend is dropped.
struct Removal {
    id: String,
    name: String,
    _keep: Option<Box<dyn Any + Send + Sync>>,
}

impl Drop for Removal {
    fn drop(&mut self) {
        let _ = std::process::Command::new("docker")
            .args(["rm", "-f", &self.id])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        crate::reaper::forget_container(&self.name);
    }
}

async fn docker_out(args: &[&str]) -> Result<String, String> {
    let out = Command::new("docker")
        .args(args)
        .output()
        .await
        .map_err(|e| format!("docker: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "docker {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

async fn docker(name: &str, bins: &Binaries, image: &str, c: Container) -> Result<Started, String> {
    let execd = bins.container()?;
    let script = format!(
        "set -e; mkdir -p /work; {}{} exec execd",
        if c.adduser {
            "(useradd -m sandbox 2>/dev/null || adduser -D sandbox); chown sandbox /work; "
        } else {
            "chmod 777 /work; "
        },
        if c.setup.is_empty() {
            String::new()
        } else {
            format!("{}; ", c.setup)
        },
    );
    let container = format!("matrix-{name}-{}-{}", std::process::id(), nonce());
    crate::reaper::watch_container(&container);
    let mut args: Vec<String> = [
        "run",
        "-d",
        "--rm",
        "--name",
        &container,
        "-p",
        "127.0.0.1::8080",
        "-v",
    ]
    .map(String::from)
    .to_vec();
    args.push(format!("{}:/usr/local/bin/execd:ro", execd.display()));
    for (k, v) in [
        ("EXECD_LISTEN", "0.0.0.0:8080"),
        ("EXECD_ROOT", "/work"),
        ("EXECD_BEARER", BEARER),
    ] {
        args.push("-e".into());
        args.push(format!("{k}={v}"));
    }
    if c.adduser {
        args.push("-e".into());
        args.push("EXECD_RUN_AS=sandbox".into());
    }
    args.extend(c.args.clone());
    args.extend([image.to_string(), "sh".into(), "-c".into(), script]);
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let id = match docker_out(&argv).await {
        Ok(id) => id,
        Err(e) => {
            crate::reaper::forget_container(&container);
            return Err(e);
        }
    };
    let guard = Removal {
        id: id.clone(),
        name: container.clone(),
        _keep: c.keep,
    };

    let started = Instant::now();
    loop {
        // A container's stderr comes back on this command's stderr.
        let logs = Command::new("docker")
            .args(["logs", &id])
            .output()
            .await
            .map(|o| {
                format!(
                    "{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                )
            })
            .unwrap_or_default();
        if logs.contains(" listening listen=") {
            break;
        }
        let running = docker_out(&["inspect", "-f", "{{.State.Running}}", &id]).await;
        if running.as_deref() != Ok("true") {
            return Err(format!(
                "container {name} exited before execd listened: {logs}"
            ));
        }
        if started.elapsed() > READY_WITHIN {
            return Err(format!("container {name}: execd did not listen: {logs}"));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let port = docker_out(&["port", &id, "8080/tcp"]).await?;
    let addr = port
        .lines()
        .next()
        .unwrap_or("")
        .replace("0.0.0.0", "127.0.0.1");
    Ok(Started {
        name: name.into(),
        url: format!("http://{addr}/exec"),
        bearer: Some(BEARER.into()),
        work: "/work".into(),
        caps: None,
        suites: None,
        vars: c.vars,
        lease: None,
        host: None,
        _guard: Box::new(guard),
    })
}

fn nonce() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
}
