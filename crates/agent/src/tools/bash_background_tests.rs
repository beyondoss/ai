//! `run_in_background` and `timeout_ms` as "hand back control": a background start returns after its
//! settle with the first output (or with the exit of a command that died at once), background jobs
//! are intended — never evicted, capped together with adopted ones, alive past the turn — and a
//! `timeout_ms` returns the call while the command keeps running; nothing kills it on a timer.
//! Linux-only, like `bash_jobs_tests`. The `run`/`serve` lifetimes are `tests/bash_background_*.rs`.

use super::super::bash_jobs::EndPolicy;
use super::*;
use agent_core::Background;

fn ok(r: Result<ToolOutput, ToolError>) -> String {
    match r {
        Ok(out) => out.text,
        Err(e) => panic!("expected success, got {e}"),
    }
}

fn alive(pid: u32) -> bool {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    let state = stat
        .rsplit_once(')')
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or("");
    !stat.is_empty() && state != "Z" && state != "X"
}

async fn wait_dead(pid: u32) -> bool {
    for _ in 0..100 {
        if !alive(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

async fn read_pid(path: &std::path::Path) -> u32 {
    for _ in 0..100 {
        if let Ok(text) = std::fs::read_to_string(path)
            && let Ok(pid) = text.trim().parse()
        {
            return pid;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("{} was never written", path.display());
}

/// A `bash` with a short settle and idle window, and the `bash_job` over it.
fn tools(settle_ms: u64, idle_ms: u64) -> (Arc<Bash>, BashJob) {
    let bash = Arc::new(
        Bash::real()
            .with_idle_yield(Some(Duration::from_millis(idle_ms)))
            .with_settle(Duration::from_millis(settle_ms)),
    );
    let job = BashJob::new(bash.clone());
    (bash, job)
}

fn background(command: &str) -> Value {
    json!({ "command": command, "run_in_background": true })
}

async fn notices_within(bash: &Bash, within: Duration) -> Option<String> {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if let Some(n) = bash.jobs().take_notices() {
            return Some(n);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    None
}

#[tokio::test]
async fn a_background_start_returns_after_the_settle_with_its_first_output() {
    let (bash, jobs) = tools(300, 30_000);
    let started = Instant::now();
    let out = ok(bash.run(background("echo serving on 8080; sleep 61")).await);
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(300) && took < Duration::from_secs(5),
        "returned at the settle: {took:?}"
    );
    assert!(out.contains("serving on 8080"), "first output: {out}");
    assert!(
        out.contains(
            "Running in the background as job 1 (it is stopped for you when the run ends)."
        ),
        "{out}"
    );
    assert!(
        out.contains("You will be told when it exits, so do not poll it or sleep."),
        "{out}"
    );
    // Peek, wait (only for something that finishes), stop: each spelled out.
    assert!(
        out.contains(
            r#"To see what it has printed since, bash_job {"job": 1, "timeout_ms": 0}; to wait for it to finish (a build or a test run, not a server), bash_job {"job": 1}; to stop it, bash_job {"job": 1, "kill": true}."#
        ),
        "{out}"
    );
    let carry = bash.jobs().carry().unwrap();
    assert!(
        carry.contains("- job 1: `echo serving on 8080; sleep 61` (running")
            && carry.contains("started in the background"),
        "{carry}"
    );
    // The output since the start is what a wait returns next, not the start's again.
    ok(jobs.run(json!({ "job": 1, "kill": true })).await);
}

#[tokio::test]
async fn a_silent_background_start_says_there_is_no_output_yet() {
    let (bash, jobs) = tools(200, 30_000);
    let out = ok(bash.run(background("sleep 62")).await);
    assert!(out.starts_with("(no output yet)\n\n"), "{out}");
    ok(jobs.run(json!({ "job": 1, "kill": true })).await);
}

#[tokio::test]
async fn a_background_command_that_dies_at_once_reports_its_exit_in_the_result() {
    let (bash, _jobs) = tools(3_000, 30_000);
    let started = Instant::now();
    let err = bash
        .run(background(
            "echo 'error: address already in use' >&2; exit 98",
        ))
        .await
        .unwrap_err();
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "an exit ends the settle early: {:?}",
        started.elapsed()
    );
    let ToolError::Execution(msg) = err else {
        panic!("a failed start is an execution error: {err:?}")
    };
    assert!(msg.contains("address already in use"), "{msg}");
    assert!(msg.contains("Command exited with code 98"), "{msg}");
    assert!(!msg.contains("job"), "no job was left: {msg}");
    assert!(bash.jobs().carry().is_none(), "the table is empty");
    assert!(
        bash.jobs().take_notices().is_none(),
        "and nothing to notice"
    );
}

#[tokio::test]
async fn a_background_job_that_exits_later_is_noticed_once() {
    let (bash, jobs) = tools(200, 30_000);
    let out = ok(bash.run(background("echo up; sleep 0.8; echo bye")).await);
    assert!(out.contains("job 1"), "{out}");
    let notice = notices_within(&bash, Duration::from_secs(10))
        .await
        .expect("a notice when it exits");
    assert!(
        notice.starts_with("[job 1 exited with code 0 after"),
        "{notice}"
    );
    assert!(
        notice.contains("bye") && !notice.contains("up\n"),
        "{notice}"
    );
    assert!(bash.jobs().take_notices().is_none(), "exactly once");
    let rest = ok(jobs.run(json!({ "job": 1 })).await);
    assert!(rest.contains("bye"), "{rest}");
}

#[tokio::test]
async fn background_jobs_are_never_evicted_by_yielded_ones() {
    let (bash, jobs) = tools(200, 200);
    ok(bash.run(background("sleep 63")).await);
    let mut last = String::new();
    for _ in 0..=MAX_YIELDED_JOBS {
        last = ok(bash.run(json!({ "command": "sleep 64" })).await);
    }
    // Jobs 2..=6 yielded; parking the fifth evicts job 2, the oldest yielded — never job 1.
    assert!(last.contains("Job 2 was stopped"), "{last}");
    assert!(
        last.contains("jobs whose calls returned early are kept running"),
        "{last}"
    );
    let carry = bash.jobs().carry().unwrap();
    assert!(
        carry.contains("- job 1: `sleep 63`"),
        "the background job stays: {carry}"
    );
    assert!(
        jobs.run(json!({ "job": 2 })).await.is_err(),
        "job 2 is gone"
    );
    bash.jobs().stop_all();
}

#[tokio::test]
async fn the_intended_cap_counts_background_and_adopted_jobs_and_refuses_before_starting() {
    let (bash, _jobs) = tools(150, 30_000);
    let half = MAX_INTENDED_JOBS / 2;
    for i in 0..half {
        ok(bash.run(background(&format!("sleep 65; echo {i}"))).await);
    }
    for i in half..MAX_INTENDED_JOBS {
        let out = ok(bash
            .run(json!({ "command": format!("sleep 66 & echo {i}") }))
            .await);
        assert!(out.contains("tracked as job"), "{out}");
    }
    let tmp = tempfile::tempdir().unwrap();
    let marker = tmp.path().join("ran");
    let err = bash
        .run(background(&format!("touch {}; sleep 67", marker.display())))
        .await
        .unwrap_err();
    let ToolError::Execution(msg) = err else {
        panic!("a refusal is an execution error: {err:?}")
    };
    assert!(
        msg.starts_with(&format!(
            "Not started: {MAX_INTENDED_JOBS} jobs started in the background or left running by \
             a command are already running ("
        )),
        "{msg}"
    );
    assert!(msg.contains("job 1 `sleep 65; echo 0`"), "{msg}");
    assert!(
        msg.contains(&format!("job {} `sleep 66 & echo {half}`", half + 1)),
        "{msg}"
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!marker.exists(), "a refused command never runs");
    // An adoption past the cap is refused against the same count.
    let adopted = ok(bash.run(json!({ "command": "sleep 68 & echo x" })).await);
    assert!(adopted.contains("It was stopped"), "{adopted}");
    bash.jobs().stop_all();
}

#[tokio::test]
async fn a_foreground_timeout_returns_the_call_and_the_command_keeps_running_to_its_exit() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (bash, _jobs) = tools(1_500, 30_000);
    let started = Instant::now();
    let out = ok(bash
        .run(json!({
            "command": format!("echo $$ > {}; echo begin; sleep 1.5; echo end", pidfile.display()),
            "timeout_ms": 300
        }))
        .await);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "returned at the timeout: {:?}",
        started.elapsed()
    );
    assert!(
        out.starts_with("begin\n") && !out.contains("\nend\n"),
        "{out}"
    );
    assert!(
        out.contains(
            "Returned after your timeout_ms of 300 ms. The command is still running as job 1"
        ),
        "{out}"
    );
    assert!(
        out.contains("it is stopped when you end your turn"),
        "{out}"
    );
    // Well past the timeout, it is still running: no timeout kills anything.
    let pid = read_pid(&pidfile).await;
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert!(alive(pid), "the command outlived its timeout_ms");
    let notice = notices_within(&bash, Duration::from_secs(10))
        .await
        .expect("its exit is noticed");
    assert!(
        notice.starts_with("[job 1 exited with code 0 after"),
        "{notice}"
    );
    assert!(notice.contains("end"), "{notice}");
}

#[tokio::test]
async fn a_timeout_ms_longer_than_the_command_changes_nothing() {
    let (bash, _jobs) = tools(1_500, 30_000);
    let out = ok(bash
        .run(json!({ "command": "echo quick", "timeout_ms": 60_000 }))
        .await);
    assert_eq!(out, "quick\n");
    assert!(bash.jobs().carry().is_none());
}

#[tokio::test]
async fn the_end_of_a_turn_keeps_background_jobs_and_stop_all_stops_them() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (bash, _jobs) = tools(200, 200);
    ok(bash
        .run(background(&format!(
            "sh -c 'echo $$ > {}; exec sleep 69'",
            pidfile.display()
        )))
        .await);
    ok(bash.run(json!({ "command": "sleep 70" })).await);
    let stopped: Vec<(u64, JobKind)> = bash
        .jobs()
        .end_turn()
        .iter()
        .map(|s| (s.job, s.kind))
        .collect();
    assert_eq!(stopped, vec![(2, JobKind::Yielded)], "only the yielded job");
    let pid = read_pid(&pidfile).await;
    assert!(alive(pid), "the background job lives past the turn");
    let stopped = bash.jobs().stop_all();
    assert_eq!(stopped.len(), 1);
    assert_eq!(stopped[0].kind, JobKind::Background);
    assert!(wait_dead(pid).await, "stop_all stops it, process and all");
}

#[tokio::test]
async fn the_reminder_says_what_ending_does_to_a_background_job_per_host() {
    let (bash, _jobs) = tools(200, 200);
    ok(bash.run(json!({ "command": "sleep 71" })).await);
    ok(bash.run(background("sleep 72")).await);
    let run = bash.jobs().stop_reminder().expect("running jobs");
    assert!(
        run.contains("- job 2: `sleep 72` (running") && run.contains("started in the background)"),
        "{run}"
    );
    assert!(
        run.contains(
            "If you end your turn now, jobs 1, 2 are stopped for you (no need to kill them). If \
             you still need a job's result"
        ),
        "{run}"
    );
    bash.jobs().set_policy(EndPolicy::KeepRunning);
    // Under serve a background job outlives the turn and its exit wakes the session: no reminder.
    ok(bash.run(background("sleep 73")).await);
    assert!(
        bash.jobs().stop_reminder().is_none(),
        "an intended job needs no reminder"
    );
    // A yielded job is stopped with the turn: the reminder names only it.
    ok(bash.run(json!({ "command": "sleep 74" })).await);
    let serve = bash.jobs().stop_reminder().expect("a new yielded job");
    assert!(
        serve.contains("- job 4: `sleep 74`")
            && !serve.contains("- job 3:")
            && !serve.contains("- job 2:"),
        "{serve}"
    );
    assert!(
        serve.contains(
            "If you end your turn now, jobs 1, 4 are stopped for you (no need to kill them)."
        ),
        "{serve}"
    );
    bash.jobs().stop_all();
}

#[tokio::test]
async fn a_background_start_in_serve_says_it_outlives_the_turn() {
    let (bash, jobs) = tools(200, 30_000);
    bash.jobs().set_policy(EndPolicy::KeepRunning);
    let out = ok(bash.run(background("sleep 74")).await);
    assert!(
        out.contains("Running in the background as job 1 (it keeps running after your turn ends)."),
        "{out}"
    );
    ok(jobs.run(json!({ "job": 1, "kill": true })).await);
}

#[tokio::test]
async fn a_wait_on_a_background_job_that_yields_keeps_it_a_background_job() {
    let (bash, jobs) = tools(200, 200);
    ok(bash.run(background("sleep 75")).await);
    let waited = ok(jobs.run(json!({ "job": 1 })).await);
    assert!(waited.contains("No output for"), "{waited}");
    assert!(
        waited.contains("s so far; it is stopped for you when the run ends)"),
        "its lifetime is its kind's, not a yielded job's: {waited}"
    );
    // Still intended: the end of the turn keeps it.
    assert!(bash.jobs().end_turn().is_empty());
    let carry = bash.jobs().carry().unwrap();
    assert!(carry.contains("started in the background"), "{carry}");
    bash.jobs().stop_all();
}

#[tokio::test]
async fn a_wait_on_an_adopted_job_that_yields_does_not_announce_the_adoption_again() {
    let (bash, jobs) = tools(200, 200);
    let out = ok(bash.run(json!({ "command": "sleep 76 & echo bg" })).await);
    assert!(out.contains("tracked as job 1"), "{out}");
    let waited = ok(jobs.run(json!({ "job": 1 })).await);
    assert!(waited.contains("No output for"), "{waited}");
    assert!(
        bash.jobs().take_notices().is_none(),
        "the adoption was its own call's news, not a notice"
    );
    bash.jobs().stop_all();
}

#[tokio::test]
async fn run_in_background_must_be_a_boolean() {
    let (bash, _jobs) = tools(200, 30_000);
    let err = bash
        .run(json!({ "command": "echo hi", "run_in_background": "soon" }))
        .await
        .unwrap_err();
    assert!(matches!(err, ToolError::InvalidInput(_)), "{err:?}");
    // `false` is the foreground call it always was.
    let out = ok(bash
        .run(json!({ "command": "echo hi", "run_in_background": false }))
        .await);
    assert_eq!(out, "hi\n");
}

// `bash_job`'s `timeout_ms`: the same meaning as on `bash` (when this call returns; the job keeps
// running), plus 0, a peek. The background A/B's waits on servers and watchers wanted the latest
// output, not an exit, and each blocked for the whole silence window.

#[tokio::test]
async fn a_peek_returns_at_once_with_what_the_job_printed_since_the_last_report() {
    let (bash, jobs) = tools(200, 30_000);
    let out = ok(bash
        .run(background(
            "echo warming; sleep 0.3; echo ready on 8080; sleep 77",
        ))
        .await);
    assert!(out.contains("warming"), "{out}");
    tokio::time::sleep(Duration::from_millis(600)).await;
    let started = Instant::now();
    // Luna fills every field: `kill: false` rides along.
    let peek = ok(jobs
        .run(json!({ "job": 1, "kill": false, "timeout_ms": 0 }))
        .await);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a peek does not wait: {:?}",
        started.elapsed()
    );
    assert!(peek.starts_with("ready on 8080\n"), "{peek}");
    assert!(!peek.contains("warming"), "already reported: {peek}");
    // Nothing to explain about why it returned: that it still runs, and how to stop it.
    assert!(peek.contains("\n\nJob 1 is still running ("), "{peek}");
    assert!(
        peek.ends_with(
            "it is stopped for you when the run ends). You will be told when it exits, so do not \
             poll it. bash_job {\"job\": 1, \"kill\": true} stops it."
        ),
        "{peek}"
    );
    // A second peek has nothing new; the job is still in the table, still background.
    let again = ok(jobs.run(json!({ "job": 1, "timeout_ms": 0 })).await);
    assert!(again.starts_with("Job 1 is still running"), "{again}");
    assert!(bash.jobs().end_turn().is_empty(), "still an intended job");
    ok(jobs.run(json!({ "job": 1, "kill": true })).await);
}

#[tokio::test]
async fn timeout_ms_bounds_a_wait_on_a_job_that_never_exits() {
    // A 30 s silence window: only the timeout can end this wait.
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (bash, jobs) = tools(200, 30_000);
    ok(bash
        .run(background(&format!(
            "sh -c 'echo $$ > {}; exec sleep 78'",
            pidfile.display()
        )))
        .await);
    let started = Instant::now();
    let waited = ok(jobs.run(json!({ "job": 1, "timeout_ms": 400 })).await);
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(400) && took < Duration::from_secs(3),
        "returned at the timeout: {took:?}"
    );
    assert!(
        waited.starts_with("Returned after your timeout_ms of 400 ms. Job 1 is still running"),
        "{waited}"
    );
    let pid = read_pid(&pidfile).await;
    assert!(alive(pid), "a timeout never stops the job");
    ok(jobs
        .run(json!({ "job": 1, "kill": true, "timeout_ms": 0 }))
        .await);
    assert!(
        wait_dead(pid).await,
        "kill wins over a timeout_ms sent with it"
    );
}

#[tokio::test]
async fn a_wait_with_timeout_ms_still_returns_the_final_result_of_a_job_that_exits_first() {
    let (bash, jobs) = tools(200, 30_000);
    ok(bash
        .run(background("echo up; sleep 0.6; echo built; exit 4"))
        .await);
    let started = Instant::now();
    let err = jobs
        .run(json!({ "job": 1, "timeout_ms": 20_000 }))
        .await
        .unwrap_err();
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "returned at the exit, not the timeout"
    );
    let ToolError::Execution(done) = err else {
        panic!("a non-zero exit is an execution error: {err:?}")
    };
    assert!(done.contains("built"), "{done}");
    assert!(done.contains("Command exited with code 4"), "{done}");
    // A peek that reaches an ended job returns its final result too.
    ok(bash.run(background("echo a; sleep 0.3; echo b")).await);
    tokio::time::sleep(Duration::from_millis(800)).await;
    let peek = ok(jobs.run(json!({ "job": 2, "timeout_ms": 0 })).await);
    assert_eq!(peek, "b\n", "the final result, nothing about a job");
}

#[tokio::test]
async fn a_bad_bash_job_timeout_ms_is_refused_before_the_job_is_touched() {
    let (bash, jobs) = tools(200, 30_000);
    ok(bash.run(background("sleep 79")).await);
    for (bad, says) in [
        (json!(-1), "must be a non-negative integer"),
        (json!(1.5), "must be a non-negative integer"),
        (json!("soon"), "must be a non-negative integer"),
        (json!(2_147_483_648_u64), "must be at most 2147483647"),
    ] {
        let err = jobs
            .run(json!({ "job": 1, "timeout_ms": bad }))
            .await
            .unwrap_err();
        let ToolError::InvalidInput(msg) = err else {
            panic!("a refusal the model must act on: {err:?}")
        };
        assert!(msg.contains(says), "{msg}");
    }
    let carry = bash.jobs().carry().unwrap();
    assert!(
        carry.contains("- job 1: `sleep 79` (running"),
        "a refused call leaves the job running: {carry}"
    );
    ok(jobs.run(json!({ "job": 1, "kill": true })).await);
}
