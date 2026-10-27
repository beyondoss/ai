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

use beyond_test_support::exec_mock::{ExecMock, MockOptions, Protocol};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt as _, BufReader};
use tokio::process::Command;

const BEARER: &str = "conformance-secret";

/// How long a started endpoint has to say it is listening.
const READY_WITHIN: Duration = Duration::from_secs(60);

/// Every backend name, and the groups that expand to several.
pub const GROUPS: &[(&str, &[&str])] = &[
    (
        "local",
        &[
            "local-plain",
            "local-bearer",
            "local-no-stdin",
            "local-grep",
            "local-busybox-grep",
        ],
    ),
    (
        "fault",
        &[
            "fault-no-stdin",
            "fault-v11",
            "fault-v12-text",
            "fault-baseline-jobs",
        ],
    ),
];
pub const SINGLES: &[&str] = &["alpine", "debian-runas", "readonly", "tinydisk"];

/// Expand group names and `*` globs to backend names, in a stable order, without duplicates.
pub fn expand(requested: &[String]) -> Result<Vec<String>, String> {
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
    Ok(out)
}

pub struct Started {
    pub name: String,
    pub url: String,
    pub bearer: Option<String>,
    pub work: String,
    pub jobs_root: String,
    /// `CONFORMANCE_CAPS`: probe results the driver is told to believe.
    pub caps: Option<Value>,
    /// The suites this backend runs; `None` is all of them.
    pub suites: Option<&'static [&'static str]>,
    pub vars: BTreeMap<String, String>,
    _guard: Box<dyn Any + Send + Sync>,
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
        "local-no-stdin" => local(name, bins, false, Some(json!({"stdin": false}))).await,
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
        "fault-no-stdin" => {
            // A v1 endpoint: ignores `stdin_base64`, so the probe finds no stdin by itself.
            fault(
                name,
                MockOptions {
                    honors_stdin: false,
                    honors_output_base64: false,
                    ..MockOptions::default()
                },
                Some(&["files", "errors", "helpers", "exec-bytes"]),
            )
            .await
        }
        "fault-v11" => {
            // An endpoint that never learned streaming: `stream` is an unknown field to it.
            fault(
                name,
                MockOptions {
                    protocol: Protocol::V11,
                    honors_output_base64: false,
                    ..MockOptions::default()
                },
                Some(&["files", "errors", "helpers", "exec-bytes"]),
            )
            .await
        }
        "fault-v12-text" => {
            // Streams, but predates `output_base64`: raw exec is exact over a stream.
            fault(
                name,
                MockOptions {
                    honors_output_base64: false,
                    ..MockOptions::default()
                },
                Some(&["exec-bytes"]),
            )
            .await
        }
        "fault-baseline-jobs" => {
            // Native jobs off: plain exec only, so jobs are the baseline.
            fault(
                name,
                MockOptions {
                    jobs: false,
                    ..MockOptions::default()
                },
                None,
            )
            .await
        }
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
    _leader: Option<crate::reaper::Leader>,
    _child: tokio::process::Child,
    _dir: tempfile::TempDir,
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
    let mut c = Command::new(execd);
    c.env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        // port-0: execd binds it, keeps it, and announces the port it got (read by `listening`).
        .env("EXECD_LISTEN", "127.0.0.1:0")
        .env("EXECD_ROOT", &work)
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
    Ok(Started {
        name: name.into(),
        url: format!("http://{addr}/exec"),
        bearer: bearer.then(|| BEARER.to_string()),
        work: work.to_string_lossy().into_owned(),
        jobs_root: work.join(".jobs").to_string_lossy().into_owned(),
        caps,
        suites: None,
        vars: BTreeMap::new(),
        _guard: Box::new(Local {
            _leader: leader,
            _child: child,
            _dir: dir,
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

// ---- in-process fault-injecting endpoint -------------------------------------------------------

async fn fault(
    name: &str,
    options: MockOptions,
    suites: Option<&'static [&'static str]>,
) -> Result<Started, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let work = dir.path().canonicalize().map_err(|e| e.to_string())?;
    let mock = ExecMock::start_with(&work, options).await;
    Ok(Started {
        name: name.into(),
        url: mock.url.clone(),
        bearer: None,
        work: work.to_string_lossy().into_owned(),
        jobs_root: work.join(".jobs").to_string_lossy().into_owned(),
        caps: None,
        suites,
        vars: BTreeMap::new(),
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
        jobs_root: "/tmp/beyond-jobs".into(),
        caps: None,
        suites: None,
        vars: c.vars,
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
