//! The `beyond` backend: an existing Beyond sandbox, reached through the Beyond edge.
//!
//! Opt-in: never in a group or a glob, and needs `BEYOND_EXEC_URL` (the `exec_url` of
//! `POST /sandboxes/{id}/exec-token`) and `BEYOND_EXEC_TOKEN_CMD`, a shell command whose stdout is a
//! fresh `bagt_v1` bearer. The harness does not own the sandbox: it does not create or delete it, so
//! the guard is empty. The command is run once per backend start; the bearer reaches the drivers as
//! `CONFORMANCE_BEARER`, and never appears in a log line. `BEYOND_EXEC_WORK` is a directory the
//! sandbox user may write (default `/home/user/work`).
use std::collections::BTreeMap;
use std::process::Stdio;

use tokio::process::Command;

use crate::backends::Started;

const DEFAULT_WORK: &str = "/home/user/work";

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

/// The variables the backend needs, or the one that is missing.
pub fn configured() -> Result<(), String> {
    for k in ["BEYOND_EXEC_URL", "BEYOND_EXEC_TOKEN_CMD"] {
        if env(k).is_none() {
            return Err(format!("backend beyond needs {k} in the environment"));
        }
    }
    Ok(())
}

pub async fn start(name: &str) -> Result<Started, String> {
    configured()?;
    let url = env("BEYOND_EXEC_URL").expect("checked");
    let cmd = env("BEYOND_EXEC_TOKEN_CMD").expect("checked");
    let out = Command::new("sh")
        .args(["-c", &cmd])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| format!("beyond: token command: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "beyond: the token command exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let bearer = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if bearer.is_empty() {
        return Err("beyond: the token command printed nothing".into());
    }
    eprintln!("beyond: using {url}");
    Ok(Started {
        name: name.into(),
        url,
        bearer: Some(bearer),
        work: env("BEYOND_EXEC_WORK").unwrap_or_else(|| DEFAULT_WORK.into()),
        caps: None,
        suites: None,
        vars: BTreeMap::new(),
        _guard: Box::new(()),
    })
}
