//! The `e2b` backend: one real E2B microVM running the released `execd`, reached over the internet
//! through E2B's port forwarding and TLS.
//!
//! Opt-in: it is never in a group or a glob, and needs `E2B_API_KEY`. The VM is made with a hard
//! auto-kill timeout ([`LIFETIME_SECS`]); `execd` is installed in it by the release's own
//! `install-execd.sh` (which verifies the tarball's sha256), not built here. The VM is deleted when
//! the guard drops, and its id is on the reaper registry so the watchdog deletes it if the harness
//! dies first. The API key travels to `curl` as a config on stdin, never on a command line.
use std::collections::BTreeMap;
use std::io::Write as _;
use std::process::Stdio;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt as _;
use tokio::process::Command;

use crate::backends::Started;

const API: &str = "https://api.e2b.dev";
/// The VM's hard lifetime, set at create time: it dies by itself even if nothing kills it.
pub const LIFETIME_SECS: u64 = 1500;
/// The port `execd` listens on in the VM.
const PORT: u16 = 8081;
const HOME: &str = "/home/user";
/// The release installed: its `execd-<tag>-linux-<arch>.tar.gz`, checked against its `.sha256`.
const DEFAULT_TAG: &str = "v0.1.16";
const READY_WITHIN: Duration = Duration::from_secs(90);

pub fn key() -> Option<String> {
    std::env::var("E2B_API_KEY").ok().filter(|k| !k.is_empty())
}

/// `curl` with headers handed over as a config on stdin, so no secret is in a process listing.
async fn curl(
    headers: &[(&str, &str)],
    args: &[&str],
    stdin_body: Option<&std::path::Path>,
) -> Result<Vec<u8>, String> {
    let mut cfg = String::new();
    for (k, v) in headers {
        cfg.push_str(&format!("header = \"{k}: {v}\"\n"));
    }
    let mut c = Command::new("curl");
    c.args(["-sS", "-m", "120", "-K", "-"])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(p) = stdin_body {
        c.args(["--data-binary"]).arg(format!("@{}", p.display()));
    }
    let mut child = c.spawn().map_err(|e| format!("curl: {e}"))?;
    let mut si = child.stdin.take().expect("piped");
    si.write_all(cfg.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    drop(si);
    let out = child
        .wait_with_output()
        .await
        .map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "curl {}: {}",
            args.last().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

struct Vm {
    id: String,
    envd_token: String,
    key: String,
}

impl Drop for Vm {
    fn drop(&mut self) {
        delete(&self.key, &self.id);
        crate::reaper::forget_sandbox(&self.id);
    }
}

/// Synchronous so it runs from `Drop`. Idempotent: a VM already gone answers 404.
fn delete(key: &str, id: &str) {
    let Ok(mut c) = std::process::Command::new("curl")
        .args([
            "-sS",
            "-m",
            "30",
            "-K",
            "-",
            "-o",
            "/dev/null",
            "-X",
            "DELETE",
        ])
        .arg(format!("{API}/sandboxes/{id}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return;
    };
    if let Some(mut si) = c.stdin.take() {
        let _ = writeln!(si, "header = \"X-API-Key: {key}\"");
    }
    let _ = c.wait();
}

impl Vm {
    /// Run a script in the VM through envd's process API (Connect, server-streaming). Returns
    /// (exit code, stdout, stderr).
    async fn run(&self, script: &str, envs: Value) -> Result<(i64, String, String), String> {
        let req = serde_json::to_vec(
            &json!({"process": {"cmd": "/bin/bash", "args": ["-c", script], "envs": envs}}),
        )
        .map_err(|e| e.to_string())?;
        let mut framed = vec![0u8];
        framed.extend((req.len() as u32).to_be_bytes());
        framed.extend(req);
        let mut f = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
        f.write_all(&framed).map_err(|e| e.to_string())?;
        let url = format!("https://49983-{}.e2b.app/process.Process/Start", self.id);
        let body = curl(
            &[
                ("X-Access-Token", &self.envd_token),
                ("Content-Type", "application/connect+json"),
                ("Connect-Protocol-Version", "1"),
            ],
            &["-X", "POST", &url],
            Some(f.path()),
        )
        .await?;
        parse_frames(&body)
    }
}

/// Connect envelopes: flag (1 byte), length (4, big-endian), JSON.
fn parse_frames(mut b: &[u8]) -> Result<(i64, String, String), String> {
    let (mut out, mut err, mut code) = (Vec::new(), Vec::new(), None);
    let dec = |v: &Value, k: &str| {
        v.get(k)
            .and_then(Value::as_str)
            .and_then(|s| base64::engine::general_purpose::STANDARD.decode(s).ok())
    };
    while b.len() >= 5 {
        let len = u32::from_be_bytes([b[1], b[2], b[3], b[4]]) as usize;
        if b.len() < 5 + len {
            break;
        }
        let v: Value = serde_json::from_slice(&b[5..5 + len]).unwrap_or(Value::Null);
        b = &b[5 + len..];
        if let Some(e) = v.get("error") {
            return Err(format!("envd: {e}"));
        }
        let ev = &v["event"];
        if let Some(d) = dec(&ev["data"], "stdout") {
            out.extend(d);
        }
        if let Some(d) = dec(&ev["data"], "stderr") {
            err.extend(d);
        }
        if ev.get("end").is_some() {
            code = Some(ev["end"]["exitCode"].as_i64().unwrap_or(0));
        }
    }
    let code = code.ok_or("envd: the process stream ended without an exit status")?;
    Ok((
        code,
        String::from_utf8_lossy(&out).into_owned(),
        String::from_utf8_lossy(&err).into_owned(),
    ))
}

pub async fn start(name: &str) -> Result<Started, String> {
    let key = key().ok_or("the e2b backend needs E2B_API_KEY in the environment")?;
    let tag = std::env::var("E2B_EXECD_VERSION").unwrap_or_else(|_| DEFAULT_TAG.into());
    let t0 = Instant::now();
    let resp = curl(
        &[("X-API-Key", &key), ("Content-Type", "application/json")],
        &[
            "-X",
            "POST",
            &format!("{API}/sandboxes"),
            "-d",
            &json!({
                "templateID": std::env::var("E2B_TEMPLATE").unwrap_or_else(|_| "base".into()),
                "timeout": LIFETIME_SECS,
                "secure": true,
                "allow_internet_access": true,
                "metadata": {"beyond-conformance": std::process::id().to_string()},
            })
            .to_string(),
        ],
        None,
    )
    .await?;
    let resp: Value = serde_json::from_slice(&resp).map_err(|e| e.to_string())?;
    let (Some(id), Some(tok)) = (resp["sandboxID"].as_str(), resp["envdAccessToken"].as_str())
    else {
        return Err(format!(
            "e2b: the sandbox was not created: {}",
            resp.to_string().chars().take(300).collect::<String>()
        ));
    };
    // From here the VM costs money until it is deleted: the guard owns that.
    crate::reaper::watch_sandbox(id);
    let vm = Vm {
        id: id.into(),
        envd_token: tok.into(),
        key,
    };
    eprintln!(
        "e2b: sandbox {} created ({} s lifetime)",
        vm.id, LIFETIME_SECS
    );

    let bearer = {
        use std::fmt::Write as _;
        let mut raw = [0u8; 24];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut raw))
            .map_err(|e| e.to_string())?;
        raw.iter().fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
    };
    let install = format!(
        "set -e; mkdir -p {HOME}/bin {HOME}/work\n\
         curl -fsSL --retry 3 https://github.com/beyondoss/ai/releases/download/{tag}/install-execd.sh \
           | EXECD_VERSION={tag} EXECD_INSTALL_DIR={HOME}/bin sh\n\
         sha256sum {HOME}/bin/execd"
    );
    let mut last = String::new();
    let mut installed = false;
    for _ in 0..30 {
        match vm.run(&install, json!({})).await {
            Ok((0, out, _)) => {
                eprintln!("e2b: {}", out.trim().replace('\n', "; "));
                installed = true;
                break;
            }
            Ok((c, out, err)) => {
                return Err(format!("e2b: install-execd exited {c}: {out}{err}"));
            }
            Err(e) => last = e,
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    if !installed {
        return Err(format!("e2b: envd never accepted a command: {last}"));
    }
    // execd runs as root (the VM's passwordless sudo) and drops to the VM's user for the commands it runs.
    let launch = format!(
        "sudo env -u SUDO_COMMAND -u SUDO_USER -u SUDO_UID -u SUDO_GID EXECD_LISTEN=0.0.0.0:{PORT} \
         EXECD_ROOT={HOME}/work EXECD_HOME={HOME} EXECD_RUN_AS=user EXECD_BEARER=\"$BEARER\" \
         setsid nohup {HOME}/bin/execd >/tmp/execd.log 2>&1 < /dev/null &\nsleep 1"
    );
    let (c, o, e) = vm.run(&launch, json!({"BEARER": bearer})).await?;
    if c != 0 {
        return Err(format!("e2b: starting execd exited {c}: {o}{e}"));
    }
    let url = format!("https://{PORT}-{}.e2b.app/exec", vm.id);
    // Up when the public URL answers 401 to a caller without the bearer.
    let t = Instant::now();
    loop {
        let code = curl(
            &[],
            &[
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "-m",
                "5",
                "-X",
                "POST",
                &url,
                "-d",
                "{}",
            ],
            None,
        )
        .await
        .unwrap_or_default();
        if code == b"401" {
            break;
        }
        if t.elapsed() > READY_WITHIN {
            let log = vm
                .run("cat /tmp/execd.log", json!({}))
                .await
                .map(|r| r.1)
                .unwrap_or_default();
            return Err(format!(
                "e2b: execd never answered 401 at {url} (last HTTP {}): {log}",
                String::from_utf8_lossy(&code)
            ));
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    if let Ok((_, log, _)) = vm.run("grep -m1 listening /tmp/execd.log", json!({})).await {
        eprintln!("e2b: {}", log.trim());
    }
    eprintln!("e2b: ready in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(Started {
        name: name.into(),
        url,
        bearer: Some(bearer),
        work: format!("{HOME}/work"),
        caps: None,
        suites: None,
        vars: BTreeMap::new(),
        _guard: Box::new(vm),
    })
}
