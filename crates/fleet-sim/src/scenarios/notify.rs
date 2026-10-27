//! exec v1.4 job-end callbacks across a fleet (FLEET.md, "Job-end callbacks", E1–E7).
//!
//! Two kinds of scenario. The edge's own rules (E1, E2, E4, E5, E6) are graded against stand-in
//! replicas that record every byte the front forwards and answer whatever the scenario asks, because
//! a real replica cannot show that a request was forwarded unchanged, or answer `411` on demand.
//! Everything about where a callback ends up (E3, E7, and the takeover, ring-change and
//! closed-session scenarios) runs real `serve --service` replicas started with the front's origin
//! as `--notify-origin`, over both kinds of v1.4 sandbox: one that keeps jobs natively and calls
//! back itself, and the baseline, whose `setsid` supervisor calls back with `curl`.
//!
//! The job in every real-replica scenario waits on a flag file the scenario creates, so it ends
//! exactly when the scenario says, however long a takeover on a shared filesystem takes.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beyond_test_support::exec_mock::{ExecMock, MockOptions};
use beyond_test_support::{spawn_model_server_routed, turn_text, turn_tool_use};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{ATTACHED_SKIP, Fleet, Notify, Outcome, replica_at};
use crate::check::{self, Finding};
use crate::edge::{Addr, Placement, Target, ring_order};
use crate::front::{self, Front};
use crate::substrate::Kind;
use crate::workload;

mod flows;
mod rules;

pub use flows::{
    job_ended_across_a_ring_change, job_ended_after_takeover, job_ended_after_the_session_closed,
    job_ended_routed_as_any_request, job_ended_token_never_logged,
};
pub use rules::{
    job_ended_answers_within_25s, job_ended_classified_exactly, job_ended_forwarded_unchanged,
    job_ended_rate_limited_per_session, job_ended_status_passes_through,
};

/// Send `raw` to the front; its whole answer, and how long it took. Bounded well past E4's 25 s.
async fn send(authority: &str, raw: &[u8]) -> Result<(Vec<u8>, Duration), String> {
    let began = Instant::now();
    let fut = async {
        let mut s = tokio::net::TcpStream::connect(authority)
            .await
            .map_err(|e| format!("connect {authority}: {e}"))?;
        s.write_all(raw).await.map_err(|e| format!("write: {e}"))?;
        let mut out = Vec::new();
        if let Err(e) = s.read_to_end(&mut out).await
            && out.is_empty()
        {
            return Err(format!("read: {e}"));
        }
        Ok(out)
    };
    let out = tokio::time::timeout(Duration::from_secs(40), fut)
        .await
        .map_err(|_| "the front did not answer within 40 s".to_string())??;
    Ok((out, began.elapsed()))
}

/// A callback request for `target` (path and query) with `headers` as whole lines and `body`,
/// `Content-Length` added unless a header already frames the body.
fn request(method: &str, target: &str, headers: &[String], body: &str) -> Vec<u8> {
    let mut r = format!("{method} {target} HTTP/1.1\r\nHost: edge.test\r\n");
    let framed = headers.iter().any(|h| {
        let h = h.to_ascii_lowercase();
        h.starts_with("content-length:") || h.starts_with("transfer-encoding:")
    });
    for h in headers {
        r.push_str(h);
        r.push_str("\r\n");
    }
    if !framed {
        r.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    r.push_str("\r\n");
    r.push_str(body);
    r.into_bytes()
}

const FAKE_TOKEN: &str = "bjn_v1.5d1c0a39.Jbnotarealmac";
const BODY: &str = r#"{"type":"job_ended","scope":"s","id":"0123456789abcdef"}"#;

fn bearer(token: &str) -> String {
    format!("Authorization: Bearer {token}")
}

fn agent_path(session: &str, extra: &str) -> String {
    format!("/_beyond/agent?session_id={session}&tenant=t1{extra}")
}

fn report(name: &str, findings: &[Finding]) -> Outcome {
    if check::report(name, findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

fn finding(claim: &'static str, ok: bool, detail: impl Into<String>) -> Finding {
    Finding {
        claim,
        ok,
        detail: detail.into(),
    }
}

// ---------------------------------------------------------------------------------------------
// Real replicas: where a callback ends up.
// ---------------------------------------------------------------------------------------------

/// Flag files that end the scenarios' jobs. Dropping it ends every job still waiting, so a failed
/// scenario does not leave one running on the host.
pub(super) struct Flags {
    dir: tempfile::TempDir,
    tags: Vec<String>,
}

impl Flags {
    fn path(&self, tag: &str) -> PathBuf {
        self.dir.path().join(tag)
    }

    fn raise(&self, tag: &str) -> Result<(), String> {
        std::fs::write(self.path(tag), b"").map_err(|e| format!("flag {tag}: {e}"))
    }
}

impl Drop for Flags {
    fn drop(&mut self) {
        for t in &self.tags {
            let _ = std::fs::write(self.dir.path().join(t), b"");
        }
    }
}

/// The model's side: `start-job-<tag>` starts a background job that waits for its flag and then
/// prints `fin-<tag>`; `check-<tag>` runs a plain command (a real use of the sandbox). Checked in
/// order, the latest stage of a conversation first.
fn routes(flags: &Flags) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for tag in &flags.tags {
        let flag = flags.path(tag);
        out.push((
            format!("\"tool_use_id\":\"ck-{tag}\""),
            turn_text("checked"),
        ));
        out.push((
            format!("check-{tag}"),
            turn_tool_use(
                &format!("ck-{tag}"),
                "bash",
                &json!({ "command": "echo checking" }).to_string(),
            ),
        ));
        out.push((
            format!("\"tool_use_id\":\"nj-{tag}\""),
            turn_text("started"),
        ));
        out.push((
            format!("start-job-{tag}"),
            turn_tool_use(
                &format!("nj-{tag}"),
                "bash",
                &json!({
                    "command": format!(
                        "while [ ! -e '{}' ]; do sleep 0.2; done; printf 'fin-%s\\n' {tag}",
                        flag.display()
                    ),
                    "run_in_background": true,
                })
                .to_string(),
            ),
        ));
    }
    out
}

/// The two v1.4 sandboxes: one that keeps jobs natively and calls back itself, and one without
/// native jobs whose notify is the baseline's (`curl` from the job's supervisor).
const SANDBOXES: [(&str, bool); 2] = [
    ("native (v1.3 + notify)", true),
    ("baseline (v1.2 + notify)", false),
];

type Sandboxes = Vec<(&'static str, ExecMock, tempfile::TempDir)>;

/// A fleet of `replicas` that offers notify through its front, whose model follows [`routes`] for
/// `tags`, and the two sandboxes.
pub(super) async fn notify_fleet(
    kind: Kind,
    history: &Path,
    replicas: usize,
    idle_timeout: Option<u64>,
    tags: &[String],
) -> Result<(Fleet, Sandboxes, Flags), String> {
    if super::ATTACHED.get().is_some() {
        return Err(format!(
            "{ATTACHED_SKIP}the scenario needs replicas started with this process's callback front \
             as --notify-origin, and a model that follows its own routes"
        ));
    }
    let flags = Flags {
        dir: tempfile::tempdir().map_err(|e| format!("flags: {e}"))?,
        tags: tags.to_vec(),
    };
    let (gateway_url, _bodies) = spawn_model_server_routed(routes(&flags), turn_text("ok"));
    let fleet = Fleet::start_against_with(
        kind,
        replicas,
        1,
        history,
        None,
        false,
        gateway_url,
        &Notify {
            front: true,
            idle_timeout,
        },
    )
    .await?;
    let mut sandboxes = Vec::new();
    for (name, jobs) in SANDBOXES {
        let dir = tempfile::tempdir().map_err(|e| format!("sandbox: {e}"))?;
        let mock = ExecMock::start_with(
            dir.path(),
            MockOptions {
                jobs,
                ..MockOptions::notifying()
            },
        )
        .await;
        sandboxes.push((name, mock, dir));
    }
    Ok((fleet, sandboxes, flags))
}

/// The first session id with `stem` that `targets` hash to `want`.
fn session_on(targets: &[Target], want: &str, stem: &str) -> Result<String, String> {
    (0..100_000)
        .map(|i| format!("s1.{stem}-{i}"))
        .find(|id| {
            ring_order(targets, id)
                .first()
                .is_some_and(|t| t.name == want)
        })
        .ok_or_else(|| format!("no session id with stem {stem} hashes to {want}"))
}

fn targets_named(fleet: &Fleet, names: &[&str]) -> Vec<Target> {
    fleet
        .replicas
        .iter()
        .filter(|r| names.contains(&r.name.as_str()))
        .map(|r| Target {
            name: r.name.clone(),
            addr: r.addr.clone(),
        })
        .collect()
}

/// Put every running replica back in the ring.
fn retarget_all(fleet: &mut Fleet) {
    let all = fleet
        .replicas
        .iter()
        .filter(|r| r.is_running())
        .map(|r| Target {
            name: r.name.clone(),
            addr: r.addr.clone(),
        })
        .collect();
    fleet.edge.set_targets(all);
}

fn front_of(fleet: &Fleet) -> Result<&Front, String> {
    fleet
        .front
        .as_ref()
        .ok_or_else(|| "the fleet has no front".into())
}

/// The front's answered callbacks for `session`, in order: (status, by, attempts).
fn answers(fleet: &Fleet, session: &str) -> Vec<Value> {
    fleet
        .front
        .as_ref()
        .map(|f| f.log_for(session))
        .unwrap_or_default()
}

/// Wait (bounded) for the `n`th callback for `session` the front answered with `status`.
async fn nth_answer(
    fleet: &Fleet,
    session: &str,
    status: u16,
    n: usize,
    within: Duration,
) -> Result<Value, String> {
    let deadline = Instant::now() + within;
    loop {
        let hits: Vec<Value> = answers(fleet, session)
            .into_iter()
            .filter(|e| e["answered"] == status)
            .collect();
        if let Some(e) = hits.get(n - 1) {
            return Ok(e.clone());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "no callback #{n} for {session} answered {status} within {within:?}; the front \
                 answered: {}",
                summarize(&answers(fleet, session))
            ));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn count_answered(fleet: &Fleet, session: &str, status: u16) -> usize {
    answers(fleet, session)
        .iter()
        .filter(|e| e["answered"] == status)
        .count()
}

/// One line per front entry: `answered by [attempts]`.
fn summarize(entries: &[Value]) -> String {
    entries
        .iter()
        .map(|e| {
            format!(
                "{} by {} after {}",
                e["answered"],
                e["by"].as_str().unwrap_or("-"),
                e["attempts"]
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The replicas an entry's attempts reached, in order, with what each answered.
fn attempts(entry: &Value) -> Vec<(String, String)> {
    entry["attempts"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|x| {
                    (
                        x[0].as_str().unwrap_or("").to_owned(),
                        match &x[1] {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

const NOTICE: &str = "[job 1 exited with code 0 after";

/// How many times the job's end notice is in the session's transcript.
async fn notices(ws: &mut workload::Ws) -> Result<usize, String> {
    let t = Value::Array(workload::transcript(ws).await?).to_string();
    Ok(t.matches(NOTICE).count())
}

/// Wait (bounded) for the end notice, then a little longer for a second one that must not come.
async fn notices_settled(ws: &mut workload::Ws, within: Duration) -> Result<usize, String> {
    let deadline = Instant::now() + within;
    loop {
        let n = notices(ws).await?;
        if n > 0 || Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    tokio::time::sleep(Duration::from_secs(3)).await;
    notices(ws).await
}

/// Open `session` on whichever replica the edge places it on, and start the job; the client stays
/// attached. Where it was placed.
async fn open_and_start(
    fleet: &Fleet,
    session: &str,
    grant: &str,
    tag: &str,
) -> Result<(Addr, workload::Ws), String> {
    let placed = crate::edge::place(&fleet.edge, session, grant, Duration::from_secs(30)).await;
    let Placement::Served { addr, .. } = placed else {
        return Err(format!("could not place {session}: {placed:?}"));
    };
    let mut ws = workload::connect(&addr, session, grant).await?;
    workload::prompt(&mut ws, &format!("start-job-{tag}")).await?;
    Ok((addr, ws))
}

fn name_at(fleet: &Fleet, addr: &Addr) -> String {
    replica_at(fleet, addr)
        .map(|i| fleet.replicas[i].name.clone())
        .unwrap_or_else(|| addr.to_string())
}

/// Run `each` once per sandbox, prefixing its findings with the sandbox's name; a setup error
/// becomes a failed finding.
async fn per_sandbox<F>(
    fleet: &mut Fleet,
    sandboxes: &Sandboxes,
    flags: &Flags,
    claim: &'static str,
    mut each: F,
) -> Vec<Finding>
where
    F: AsyncFnMut(&mut Fleet, &ExecMock, &Flags, &str, usize) -> Result<Vec<Finding>, String>,
{
    let mut out = Vec::new();
    for (i, (name, exec, _dir)) in sandboxes.iter().enumerate() {
        match each(fleet, exec, flags, name, i).await {
            Ok(fs) => out.extend(fs.into_iter().map(|mut f| {
                f.detail = format!("{name}: {}", f.detail);
                f
            })),
            Err(e) => out.push(finding(claim, false, format!("{name}: {e}"))),
        }
        // Leave every replica up and in the ring for the next sandbox.
        for idx in 0..fleet.replicas.len() {
            if !fleet.replicas[idx].is_running()
                && let Err(e) = fleet.restart(idx).await
            {
                out.push(finding(claim, false, format!("restart: {e}")));
            }
        }
        retarget_all(fleet);
    }
    out
}

fn tags(stem: &str) -> Vec<String> {
    (0..SANDBOXES.len()).map(|i| format!("{stem}{i}")).collect()
}
