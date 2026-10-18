//! `bash` jobs the sandbox runs ([`crate::exec_endpoint::jobs`]): how the job table starts them,
//! hands them over, kills them, and finds them again on a session's next attach.
//!
//! Every remote command of a session with a job scope starts **held** in the sandbox (or kept, over
//! an endpoint that cannot stream): its stream is the attachment, so a foreground command still dies
//! with its call, its turn or its replica. The moment it becomes an intended job — parked by a
//! `run_in_background` start, or adopted when its shell exits leaving processes running — the table
//! **keeps** it under its job number ([`keep`]), and from then on the sandbox owns it: dropping the
//! job (a reaped or redeployed replica) lets go of the stream and nothing more, and only a kill, the
//! end of the session, or the sandbox's own end stops it. [`reattach`] is the other half: the next
//! replica to open the session lists the scope, adopts what still runs under its old number, delivers
//! the end of what finished meanwhile as an ordinary finish notice, and kills what nobody kept — or,
//! when its sandbox is not attached yet, defers that to the session's first use of it
//! ([`reattach_later`], [`reconcile`]), so opening a session does not wake a sleeping sandbox.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::bash_jobs::{BashJobs, EndPolicy, Job, JobKind, JobShared};
use super::exec::{CommandRunner, ExecResult, Kill};
use crate::exec_endpoint::jobs::{JobEvent, JobId, JobScope, JobSpec, JobStatus, SandboxJobs};

/// What a sandbox job's leftovers are called, when its shell exits leaving them running.
pub(crate) const SANDBOX_LEFTOVERS: &str = "processes in the sandbox";

/// How long a kill may take before it is abandoned: a round trip, the grace, and the wait for the
/// group to empty.
const KILL_LIMIT: Duration = Duration::from_secs(15);

/// Where the sandbox runs a job. Clones share `kept`.
#[derive(Clone)]
pub(crate) struct SandboxLink {
    pub store: Arc<dyn SandboxJobs>,
    pub scope: JobScope,
    pub id: JobId,
    kept: Arc<AtomicBool>,
}

impl SandboxLink {
    pub fn new(store: Arc<dyn SandboxJobs>, scope: JobScope, id: JobId, kept: bool) -> Self {
        Self {
            store,
            scope,
            id,
            kept: Arc::new(AtomicBool::new(kept)),
        }
    }

    /// The sandbox owns it: a dropped stream no longer kills it.
    pub fn kept(&self) -> bool {
        self.kept.load(Ordering::Relaxed)
    }
}

/// The sandbox and scope this table's remote commands start jobs in, when it has both and the
/// sandbox can run them.
pub(crate) async fn store_for(
    jobs: &BashJobs,
    runner: &Arc<dyn CommandRunner>,
) -> Option<(Arc<dyn SandboxJobs>, JobScope)> {
    let scope = jobs.scope()?;
    let store = runner.clone().sandbox_jobs()?;
    store.available().await.then_some((store, scope))
}

/// The [`Kill`] for a sandbox job: the `kill` operation (`SIGTERM`, the grace, `SIGKILL`), which
/// also deletes its directory. Blocking, like every [`Kill`]; the operation itself runs on the
/// runtime the job started on.
pub(crate) fn kill_hook(link: &SandboxLink) -> Kill {
    let link = link.clone();
    let runtime = tokio::runtime::Handle::try_current().ok();
    Kill::Hook(Arc::new(move || {
        let Some(runtime) = &runtime else {
            return Some(UNREACHED.to_string());
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let l = link.clone();
        runtime.spawn(async move {
            let r = l
                .store
                .kill(&l.scope, std::slice::from_ref(&l.id), true)
                .await;
            let _ = tx.send(r);
        });
        match rx.recv_timeout(KILL_LIMIT) {
            Ok(Ok(survivors)) => survivors.into_iter().next().map(|(_, d)| d),
            Ok(Err(e)) => {
                tracing::warn!(job = %link.id, error = %e, "a sandbox job's kill failed");
                Some(UNREACHED.to_string())
            }
            Err(_) => {
                tracing::warn!(job = %link.id, "a sandbox job's kill did not finish in time");
                Some(UNREACHED.to_string())
            }
        }
    }))
}

/// What a kill that never reached the sandbox left, as the model is told.
pub(crate) const UNREACHED: &str = "the job's processes in the sandbox (the kill did not reach it)";

/// A sandbox job's event, applied to the job's shared state.
pub(crate) fn on_event(shared: &JobShared, link: &SandboxLink, event: JobEvent<'_>) {
    match event {
        // Started means the output is live from here: what `bash`'s idle window measures.
        JobEvent::Started(_) => shared.track(kill_hook(link)),
        JobEvent::Output(bytes) => shared.append(bytes),
        JobEvent::Gap(n) => shared.append_gap(n),
        JobEvent::ShellExit(code) => shared.orphaned(
            &ExecResult {
                code: Some(code),
                ..Default::default()
            },
            SANDBOX_LEFTOVERS.to_string(),
        ),
    }
}

/// A job's status as `bash` reports a result.
fn status_result(code: i32) -> ExecResult {
    ExecResult {
        code: Some(code),
        ..Default::default()
    }
}

/// The runner task of a new remote command run as a sandbox job: held when the sandbox can stream
/// it, kept from the start otherwise. A held stream that breaks after the job was kept does not end
/// the job; it is followed from where the stream left off.
pub(crate) async fn run(
    shared: &Arc<JobShared>,
    link: &SandboxLink,
    spec: &JobSpec,
    held: bool,
) -> Result<ExecResult, String> {
    let events = |e: JobEvent<'_>| on_event(shared, link, e);
    if held {
        match link
            .store
            .run_held(&link.scope, &link.id, spec, &events)
            .await
        {
            Ok(code) => return Ok(status_result(code)),
            Err(e) if !link.kept() => return Err(e.to_string()),
            Err(e) => {
                tracing::info!(job = %link.id, error = %e, "a kept job's stream broke; following it by polling");
            }
        }
    } else {
        let pgid = link
            .store
            .start_kept(&link.scope, &link.id, spec)
            .await
            .map_err(|e| e.to_string())?;
        events(JobEvent::Started(pgid));
    }
    follow(shared, link, shared.offset()).await
}

/// Follow a job from `from` to its end.
async fn follow(shared: &JobShared, link: &SandboxLink, from: u64) -> Result<ExecResult, String> {
    let events = |e: JobEvent<'_>| on_event(shared, link, e);
    link.store
        .follow(&link.scope, &link.id, from, &events)
        .await
        .map(status_result)
        .map_err(|e| e.to_string())
}

/// Hand a parked intended job to the sandbox under its number: from now on it outlives its stream,
/// and — on a host that keeps jobs past the session ([`EndPolicy::KeepRunning`]) — this process.
/// The offset recorded as seen is the output the model has been shown, which is all of it so far:
/// the call that parks a job reports its output first.
pub(crate) async fn keep(
    shared: &JobShared,
    num: u64,
    kind: JobKind,
    policy: EndPolicy,
) -> Result<(), String> {
    let Some(link) = shared.sandbox() else {
        return Ok(());
    };
    link.store
        .keep(&link.scope, &link.id, num, kind_name(kind), shared.offset())
        .await
        .map_err(|e| e.to_string())?;
    link.kept.store(true, Ordering::Relaxed);
    if policy == EndPolicy::KeepRunning {
        shared.release();
    }
    Ok(())
}

/// [`keep`], from code that cannot wait for it (a notice, under the table's lock).
pub(crate) fn spawn_keep(shared: Arc<JobShared>, num: u64, kind: JobKind, policy: EndPolicy) {
    if shared.sandbox().is_none() {
        return;
    }
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    runtime.spawn(async move {
        if let Err(e) = keep(&shared, num, kind, policy).await {
            tracing::warn!(job = num, error = %e, "the sandbox could not take over an adopted job");
        }
    });
}

/// Record that the model has been shown a kept job's output up to where it is now, so a replica
/// that reattaches resumes from there. Fire and forget.
pub(crate) fn spawn_seen(shared: &JobShared) {
    let Some(link) = shared.sandbox().filter(|l| l.kept()).cloned() else {
        return;
    };
    let offset = shared.offset();
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn(async move {
            let _ = link.store.seen(&link.scope, &link.id, offset).await;
        });
    }
}

fn kind_name(kind: JobKind) -> &'static str {
    match kind {
        JobKind::Yielded => "yielded",
        JobKind::Background => "background",
        JobKind::Adopted => "adopted",
    }
}

/// Delete the directories of reported, ended jobs, one operation per sandbox scope.
pub(crate) async fn remove_all(garbage: Vec<SandboxLink>) {
    let mut groups: Vec<(SandboxLink, Vec<JobId>)> = Vec::new();
    for link in garbage {
        match groups
            .iter_mut()
            .find(|(l, _)| Arc::ptr_eq(&l.store, &link.store) && l.scope == link.scope)
        {
            Some((_, ids)) => ids.push(link.id),
            None => {
                let id = link.id.clone();
                groups.push((link, vec![id]));
            }
        }
    }
    for (link, ids) in groups {
        if let Err(e) = link.store.remove(&link.scope, &ids).await {
            tracing::warn!(error = %e, "could not delete ended sandbox jobs");
        }
    }
}

/// What [`reattach`] found.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Reattached {
    /// Kept jobs still running, adopted under their numbers.
    pub running: Vec<u64>,
    /// Kept jobs that ended while nobody was attached: their notices are due.
    pub ended: Vec<u64>,
    /// Kept jobs this host's policy stops on a reopen (`run`'s), killed.
    pub stopped: Vec<u64>,
    /// Jobs nobody kept (a foreground command whose replica went away), killed.
    pub stale: usize,
}

/// A session is opening on this table: read the scope's jobs back from the sandbox. A kept job
/// with a number is this session's: under [`EndPolicy::KeepRunning`] it is parked again under that
/// number — followed from the output the model last saw, so its notice (when it ends, or at once if
/// it already has) quotes only what is new — and under [`EndPolicy::StopAtEnd`] it is killed, as the
/// end of the run that started it would have. Anything else belongs to a turn that is over and is
/// killed. Never fails: a sandbox that cannot be read is reported and left alone.
pub async fn reattach(jobs: &Arc<BashJobs>, runner: Arc<dyn CommandRunner>) -> Reattached {
    read_back(jobs, runner, false).await.unwrap_or_default()
}

/// How long [`reconcile`] waits for the ends of the jobs it found already ended to be read: one
/// read each, from a sandbox that just answered.
const SETTLE_LIMIT: Duration = Duration::from_secs(10);

/// A reattach deferred to the session's first use of its sandbox — see [`reattach_later`].
pub struct PendingReattach {
    runner: Arc<dyn CommandRunner>,
    /// Single flight: whoever needs the jobs first reads them back; the rest wait for it.
    gate: tokio::sync::Mutex<()>,
}

/// [`reattach`], deferred: the session is opening without touching its sandbox (a sandbox sleeps
/// when idle, and reading its jobs back would wake it). The table keeps what the transcript says;
/// the reopen note calls the jobs the sandbox may still hold "not yet checked"; and [`reconcile`]
/// reads them back the first time something needs the sandbox or asks about a job.
pub fn reattach_later(jobs: &Arc<BashJobs>, runner: Arc<dyn CommandRunner>) {
    jobs.set_pending_reattach(Some(Arc::new(PendingReattach {
        runner,
        gate: tokio::sync::Mutex::new(()),
    })));
}

/// Run a deferred [`reattach`] now, if one is pending; concurrent callers share it. `Err` when the
/// sandbox could not be read — its probe failed, or the listing did — in which case it stays
/// pending, for the next use to try again.
pub async fn reconcile(jobs: &Arc<BashJobs>) -> Result<(), String> {
    let Some(pending) = jobs.pending_reattach() else {
        return Ok(());
    };
    let _flight = pending.gate.lock().await;
    // Settled (or replaced by a transcript switch) while this caller waited.
    if !jobs
        .pending_reattach()
        .is_some_and(|p| Arc::ptr_eq(&p, &pending))
    {
        return Ok(());
    }
    pending.runner.ready().await.map_err(|e| e.to_string())?;
    let mut wake = jobs.subscribe();
    let report = read_back(jobs, pending.runner.clone(), true)
        .await
        .ok_or_else(|| "the sandbox's jobs could not be listed".to_string())?;
    if report != Reattached::default() {
        tracing::info!(?report, "reattached the session's sandbox jobs");
    }
    // A job that ended while nobody was attached is read to its end on its own task. Unlike a
    // reattach at open, this one answers a caller asking right now (a `bash_job` peek at it), so
    // it waits — bounded — until those ends are in, rather than report a finished job running.
    let deadline = tokio::time::Instant::now() + SETTLE_LIMIT;
    while !report.ended.iter().all(|&id| jobs.ended_or_gone(id)) {
        if tokio::time::timeout_at(deadline, wake.changed())
            .await
            .map_or(true, |r| r.is_err())
        {
            break;
        }
    }
    if jobs
        .pending_reattach()
        .is_some_and(|p| Arc::ptr_eq(&p, &pending))
    {
        jobs.set_pending_reattach(None);
    }
    Ok(())
}

/// The most reads [`read_in`] makes, each of [`crate::exec_endpoint::jobs::READ_MAX`]: what a peek
/// shows is a tail anyway.
const READ_IN_READS: usize = 8;

/// Read a job's output from `from` up to what the sandbox holds now into `shared`, returning where
/// it got to. Best effort: a read that fails leaves the rest to the follow.
async fn read_in(shared: &JobShared, link: &SandboxLink, from: u64) -> u64 {
    let mut offset = from;
    for _ in 0..READ_IN_READS {
        let Ok(chunk) = link
            .store
            .read(
                &link.scope,
                &link.id,
                offset,
                crate::exec_endpoint::jobs::READ_MAX,
            )
            .await
        else {
            break;
        };
        if chunk.offset > offset {
            shared.append_gap(chunk.offset - offset);
            offset = chunk.offset;
        }
        if chunk.data.is_empty() {
            break;
        }
        shared.append(&chunk.data);
        offset += chunk.data.len() as u64;
        if offset >= chunk.end {
            break;
        }
    }
    offset
}

/// [`reattach`]'s body. `None` when the sandbox's jobs could not be listed (as opposed to: there
/// is no job store to list, which is an answer). `catch_up` reads each running job's new output in
/// before returning ([`read_in`]).
async fn read_back(
    jobs: &Arc<BashJobs>,
    runner: Arc<dyn CommandRunner>,
    catch_up: bool,
) -> Option<Reattached> {
    let mut report = Reattached::default();
    let Some((store, scope)) = store_for(jobs, &runner).await else {
        return Some(report);
    };
    let listed = match store.list(&scope).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(error = %e, "could not list the session's sandbox jobs");
            return None;
        }
    };
    let policy = jobs.policy();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ago = |epoch: u64| {
        Instant::now()
            .checked_sub(Duration::from_secs(now.saturating_sub(epoch)))
            .unwrap_or_else(Instant::now)
    };
    let mut kill = Vec::new();
    for j in listed {
        let num = j.num.filter(|_| j.kept);
        let Some(num) = num.filter(|_| policy == EndPolicy::KeepRunning) else {
            if let Some(n) = num
                && !j.status.ended()
            {
                report.stopped.push(n);
            } else if num.is_none() && !j.status.ended() {
                report.stale += 1;
            }
            kill.push(j.id);
            continue;
        };
        let link = SandboxLink::new(store.clone(), scope.clone(), j.id.clone(), true);
        let shared = Arc::new(JobShared::in_sandbox(link.clone(), j.seen, jobs));
        shared.release();
        let ended_at = matches!(j.status, JobStatus::Ended(_)).then(|| ago(j.ended));
        match j.status {
            JobStatus::Lost | JobStatus::Gone => {
                shared.finish_at(
                    Err(
                        "its supervisor in the sandbox is gone, so how it ended was not recorded \
                         (the sandbox may have restarted)"
                            .to_string(),
                    ),
                    ago(j.ended.max(j.started)),
                );
                report.ended.push(num);
            }
            status => {
                shared.track(kill_hook(&link));
                if let JobStatus::ShellExited(code) = status {
                    shared.orphaned(&status_result(code), SANDBOX_LEFTOVERS.to_string());
                }
                if status.ended() {
                    report.ended.push(num);
                } else {
                    report.running.push(num);
                }
            }
        }
        // A read-back that answers a caller asking right now reads a running job's new output in
        // before handing it over, so a peek at it reports what it printed while nobody was
        // attached rather than whatever the follow below has fetched so far.
        let from = if catch_up && !shared.ended() && !j.status.ended() {
            read_in(&shared, &link, j.seen).await
        } else {
            j.seen
        };
        let handle = {
            let shared = shared.clone();
            let link = link.clone();
            tokio::spawn(async move {
                if shared.ended() {
                    return;
                }
                let result = follow(&shared, &link, from).await;
                shared.untrack();
                match ended_at {
                    Some(at) => shared.finish_at(result, at),
                    None => shared.finish(result),
                }
            })
        };
        let kind = match j.kind.as_deref() {
            Some("adopted") => JobKind::Adopted,
            _ => JobKind::Background,
        };
        jobs.insert_reattached(
            num,
            Job {
                id: Some(num),
                kind,
                shared,
                handle,
                started: ago(j.started),
                // The reopen note lists it; the stop reminder need not again.
                reminded: true,
                command: j.label,
            },
        );
    }
    if !kill.is_empty()
        && let Err(e) = store.kill(&scope, &kill, true).await
    {
        tracing::warn!(error = %e, "could not stop stale sandbox jobs");
    }
    Some(report)
}

/// Kill every job of `scope` in the sandbox `runner` reaches and delete them: the session is
/// ended (a transcript switch, a delete). Best effort: a sandbox that cannot be reached is gone or
/// will be torn down with its session.
pub async fn wipe(runner: Arc<dyn CommandRunner>, scope: &JobScope) {
    // A sandbox attached lazily and not used yet is connected for this: ending a session is a
    // reason to reach it, and its jobs are the ones being ended.
    if runner.ready().await.is_err() {
        return;
    }
    let Some(store) = runner.sandbox_jobs() else {
        return;
    };
    if !store.available().await {
        return;
    }
    match store.wipe(scope).await {
        Ok(survivors) if !survivors.is_empty() => {
            tracing::warn!(
                ?survivors,
                "processes of an ended session survived its wipe"
            );
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(error = %e, "could not wipe an ended session's sandbox jobs"),
    }
}
