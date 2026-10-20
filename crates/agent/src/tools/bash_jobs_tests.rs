//! `bash`'s jobs past the idle yield: the output ceiling, adopting what `cmd &` leaves running (and
//! stopping every such process, however it detached), finish notices, the stop reminder, the
//! compaction carry, and which jobs the caps may evict. Linux-only: the process tracking reads
//! `/proc`. The loop side (where notices land in the transcript) is `agent_core`'s
//! `background_loop_tests`; the end-to-end paths (`run` stopping jobs at exit, `serve` waking) are in
//! `tests/bash_jobs_*.rs`.

use super::*;
use agent_core::Background;

fn ok(r: Result<ToolOutput, ToolError>) -> String {
    match r {
        Ok(out) => out.text,
        Err(e) => panic!("expected success, got {e}"),
    }
}

fn alive(pid: u32) -> bool {
    // Signal 0 probes; a zombie still answers, so read its state too.
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

/// The pid a backgrounded `sh -c 'echo $$ > file; exec sleep …'` wrote, once it has.
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

/// A `bash` with short windows and the `bash_job` over it, sharing one table.
fn tools(idle_ms: u64, call_ceiling_ms: u64, wait_ceiling_ms: u64) -> (Arc<Bash>, BashJob) {
    let bash = Arc::new(
        Bash::real()
            .with_idle_yield(Some(Duration::from_millis(idle_ms)))
            .with_ceilings(
                Duration::from_millis(call_ceiling_ms),
                Duration::from_millis(wait_ceiling_ms),
            ),
    );
    let job = BashJob::new(bash.clone());
    (bash, job)
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
async fn a_chatty_command_yields_at_the_ceiling() {
    let (bash, jobs) = tools(30_000, 800, 600);
    let started = Instant::now();
    let out = ok(bash
        .run(json!({ "command": "while true; do echo tick; sleep 0.05; done" }))
        .await);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "returned at the ceiling"
    );
    assert!(out.contains("tick"), "{out}");
    assert!(out.contains("Still printing after"), "says why: {out}");
    assert!(out.contains("job 1"), "{out}");
    assert!(out.contains("You will be told when it exits"), "{out}");
    assert!(out.contains("do not poll it or sleep"), "{out}");
    // A wait on it returns at its own ceiling too, still running.
    let again = ok(jobs.run(json!({ "job": 1 })).await);
    assert!(
        again.contains("Still printing after") && again.contains("Job 1 is still running"),
        "{again}"
    );
    let killed = ok(jobs.run(json!({ "job": 1, "kill": true })).await);
    assert!(
        killed.contains("Job 1 (`") && killed.contains(") killed after"),
        "{killed}"
    );
}

#[tokio::test]
async fn a_quiet_yield_says_it_went_quiet() {
    let (bash, jobs) = tools(300, 60_000, 60_000);
    let out = ok(bash.run(json!({ "command": "echo hi; sleep 63" })).await);
    assert!(out.contains("No output for"), "{out}");
    assert!(out.contains("You will be told when it exits"), "{out}");
    ok(jobs.run(json!({ "job": 1, "kill": true })).await);
}

#[tokio::test]
async fn a_backgrounded_process_holding_the_pipe_is_adopted_and_its_output_kept() {
    let (bash, jobs) = tools(30_000, 60_000, 60_000);
    let started = Instant::now();
    let out = ok(bash
        .run(json!({
            "command": "(sleep 0.6; echo late-one; sleep 0.4; echo late-two) & echo started"
        }))
        .await);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "returned at the shell's exit"
    );
    assert!(out.contains("started"), "{out}");
    // The subshell and its `sleep`.
    assert!(
        out.contains("The command exited but left 2 processes (")
            && out.contains("tracked as job 1"),
        "{out}"
    );
    assert!(
        out.contains(r#"bash_job {"job": 1, "kill": true}"#),
        "{out}"
    );
    // Its output keeps streaming into the job; its end is noticed, once.
    let notice = notices_within(&bash, Duration::from_secs(10))
        .await
        .expect("a notice when the adopted processes exit");
    assert!(notice.starts_with("[job 1 ended after"), "{notice}");
    assert!(notice.contains("left running have exited"), "{notice}");
    assert!(notice.contains("late-two"), "{notice}");
    assert!(bash.jobs().take_notices().is_none(), "exactly once");
    // The full output stays readable until bash_job reads it.
    let full = ok(jobs.run(json!({ "job": 1 })).await);
    assert!(
        full.contains("late-one") && full.contains("late-two"),
        "{full}"
    );
    assert!(
        jobs.run(json!({ "job": 1 })).await.is_err(),
        "read, so gone"
    );
}

#[tokio::test]
async fn a_backgrounded_process_with_its_output_redirected_is_adopted_too() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (bash, jobs) = tools(30_000, 60_000, 60_000);
    let out = ok(bash
        .run(json!({
            "command": format!(
                "sh -c 'echo $$ > {}; exec sleep 64' > /dev/null 2>&1 & echo started",
                pidfile.display()
            )
        }))
        .await);
    assert!(out.contains("tracked as job 1"), "{out}");
    let pid = read_pid(&pidfile).await;
    assert!(alive(pid));
    let killed = ok(jobs.run(json!({ "job": 1, "kill": true })).await);
    assert!(
        killed.contains("Job 1 (`") && killed.contains(") killed after"),
        "{killed}"
    );
    assert!(wait_dead(pid).await, "kill stops it");
}

/// Every way a command detaches from its shell: each must be adopted and stopped by a kill.
fn detaching_commands(dir: &std::path::Path) -> Vec<(String, std::path::PathBuf)> {
    let child = |name: &str, secs: u32| {
        let f = dir.join(name);
        (
            format!("sh -c 'echo $$ > {}; exec sleep {secs}'", f.display()),
            f,
        )
    };
    let (setsid, f1) = child("setsid", 65);
    let (nohup, f2) = child("nohup", 66);
    let (timeout, f3) = child("timeout", 67);
    let (double, f4) = child("double", 68);
    let (setsid_double, f5) = child("setsid-double", 69);
    vec![
        (format!("setsid {setsid} > /dev/null 2>&1 &"), f1),
        (format!("nohup {nohup} > /dev/null 2>&1 &"), f2),
        (format!("timeout 120 {timeout} > /dev/null 2>&1 &"), f3),
        (format!("( ( {double} > /dev/null 2>&1 & ) & )"), f4),
        (
            format!("( setsid {setsid_double} > /dev/null 2>&1 & ) &"),
            f5,
        ),
    ]
}

#[tokio::test]
async fn setsid_nohup_timeout_and_double_forks_are_all_stopped_by_a_kill() {
    let tmp = tempfile::tempdir().unwrap();
    let cases = detaching_commands(tmp.path());
    let script: Vec<String> = cases.iter().map(|(c, _)| c.clone()).collect();
    let (bash, jobs) = tools(30_000, 60_000, 60_000);
    let out = ok(bash
        .run(json!({ "command": format!("{}\necho launched", script.join("\n")) }))
        .await);
    assert!(out.contains("tracked as job 1"), "{out}");
    let mut pids = Vec::new();
    for (_, f) in &cases {
        pids.push(read_pid(f).await);
    }
    for &pid in &pids {
        assert!(alive(pid), "{pid} runs until stopped");
    }
    ok(jobs.run(json!({ "job": 1, "kill": true })).await);
    for (pid, (cmd, _)) in pids.iter().zip(&cases) {
        assert!(wait_dead(*pid).await, "not stopped: {cmd}");
    }
}

#[tokio::test]
async fn dropping_the_table_stops_every_detached_process() {
    let tmp = tempfile::tempdir().unwrap();
    let cases = detaching_commands(tmp.path());
    let script: Vec<String> = cases.iter().map(|(c, _)| c.clone()).collect();
    let (bash, jobs) = tools(30_000, 60_000, 60_000);
    let out = ok(bash.run(json!({ "command": script.join("\n") })).await);
    assert!(out.contains("tracked as job 1"), "{out}");
    let mut pids = Vec::new();
    for (_, f) in &cases {
        pids.push(read_pid(f).await);
    }
    // A registry dropped (a subagent returning) drops its table.
    drop((bash, jobs));
    for (pid, (cmd, _)) in pids.iter().zip(&cases) {
        assert!(wait_dead(*pid).await, "outlived its table: {cmd}");
    }
}

#[tokio::test]
async fn stop_all_stops_running_jobs_and_says_which() {
    let (bash, _jobs) = tools(200, 60_000, 60_000);
    ok(bash.run(json!({ "command": "sleep 70" })).await);
    ok(bash.run(json!({ "command": "sleep 71 & echo bg" })).await);
    let stopped = bash.jobs().stop_all();
    let ids: Vec<u64> = stopped.iter().map(|s| s.job).collect();
    assert_eq!(ids, vec![1, 2]);
    assert_eq!(stopped[0].command, "sleep 70");
    assert!(bash.jobs().carry().is_none(), "the table is empty");
}

#[tokio::test]
async fn a_parked_job_that_exits_is_noticed_once_with_its_tail() {
    let (bash, jobs) = tools(300, 60_000, 60_000);
    let out = ok(bash
        .run(json!({ "command": "echo first; sleep 0.8; seq 1 30; exit 3" }))
        .await);
    assert!(out.contains("job 1"), "{out}");
    let notice = notices_within(&bash, Duration::from_secs(10))
        .await
        .expect("a notice");
    assert!(
        notice.starts_with("[job 1 exited with code 3 after"),
        "{notice}"
    );
    assert!(
        notice.contains("`echo first; sleep 0.8; seq 1 30; exit 3`"),
        "{notice}"
    );
    assert!(
        notice.contains("\n30\n") || notice.contains("\n30"),
        "{notice}"
    );
    assert!(
        !notice.contains("\n5\n"),
        "only the last 20 lines: {notice}"
    );
    assert!(!notice.contains("\nfirst"), "already reported: {notice}");
    assert!(
        notice.contains(r#"bash_job {"job": 1} returns the full output."#),
        "{notice}"
    );
    assert!(bash.jobs().take_notices().is_none(), "exactly once");
    // `bash_job` returns everything since the last report, and the final status.
    let ToolError::Execution(full) = jobs.run(json!({ "job": 1 })).await.unwrap_err() else {
        panic!("a non-zero exit is an execution error")
    };
    assert!(
        full.contains("\n5\n") && full.contains("Command exited with code 3"),
        "{full}"
    );
}

#[tokio::test]
async fn a_job_reported_by_an_attached_wait_gets_no_notice() {
    let (bash, jobs) = tools(300, 60_000, 60_000);
    ok(bash.run(json!({ "command": "sleep 1; echo done" })).await);
    // Each wait may go quiet for one 300 ms window; the one that sees the exit reports it.
    let mut done = String::new();
    for _ in 0..20 {
        done = ok(jobs.run(json!({ "job": 1 })).await);
        if !done.contains("still running") {
            break;
        }
    }
    assert!(done.contains("done"), "{done}");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(bash.jobs().take_notices().is_none(), "no duplicate");
    assert!(!bash.jobs().has_notice());
}

#[tokio::test]
async fn the_stop_reminder_lists_running_jobs_once_and_says_what_ending_does_per_kind() {
    use super::super::bash_jobs::REMINDER_ASK;
    let (bash, jobs) = tools(200, 60_000, 60_000);
    ok(bash.run(json!({ "command": "sleep 72" })).await);
    let reminder = bash.jobs().stop_reminder().expect("a running job");
    assert!(reminder.starts_with("[background jobs]"), "{reminder}");
    assert!(
        reminder.contains("- job 1: `sleep 72` (running"),
        "{reminder}"
    );
    assert!(
        reminder.contains("its call returned early)"),
        "says why it is a job: {reminder}"
    );
    assert!(
        reminder.ends_with(&format!(
            "If you end your turn now, job 1 is stopped for you (no need to kill it). \
             {REMINDER_ASK}"
        )),
        "{reminder}"
    );
    // No unconditional "wait for one": read as an instruction, it sent models to wait out a
    // server's silence.
    assert!(!reminder.contains("Wait for one"), "{reminder}");
    assert!(bash.jobs().stop_reminder().is_none(), "once per job");
    // A wait takes the job out of the table and parks it again: that is not news.
    ok(jobs.run(json!({ "job": 1, "timeout_ms": 0 })).await);
    assert!(
        bash.jobs().stop_reminder().is_none(),
        "a re-parked job was already reminded"
    );
    // A new job is news: the next reminder lists both, each kind with its own fate.
    ok(bash.run(json!({ "command": "sleep 73 & echo bg" })).await);
    let again = bash.jobs().stop_reminder().expect("a new job");
    assert!(
        again.contains("job 1") && again.contains("job 2"),
        "{again}"
    );
    assert!(again.contains("left running by its command"), "{again}");
    assert!(
        again.ends_with(&format!(
            "If you end your turn now, jobs 1, 2 are stopped for you (no need to kill them). \
             {REMINDER_ASK}"
        )),
        "{again}"
    );
    bash.jobs()
        .set_policy(super::super::bash_jobs::EndPolicy::KeepRunning);
    // Under serve a job the command left running outlives the turn and its exit wakes the
    // session: it earns no reminder.
    ok(bash.run(json!({ "command": "sleep 74 & echo bg" })).await);
    assert!(
        bash.jobs().stop_reminder().is_none(),
        "an intended job needs no reminder"
    );
    // A yielded job is stopped with the turn: the reminder lists the yielded jobs, not the intended ones.
    ok(bash.run(json!({ "command": "sleep 75" })).await);
    let serve = bash.jobs().stop_reminder().expect("a new yielded job");
    assert!(
        serve.contains("- job 4: `sleep 75`")
            && !serve.contains("- job 3:")
            && !serve.contains("- job 2:"),
        "{serve}"
    );
    assert!(
        serve.ends_with(&format!(
            "If you end your turn now, jobs 1, 4 are stopped for you (no need to kill them). \
             {REMINDER_ASK}"
        )),
        "{serve}"
    );
    bash.jobs().stop_all();
}

#[tokio::test]
async fn the_end_of_a_turn_stops_yielded_jobs_and_keeps_intended_ones() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (bash, jobs) = tools(200, 60_000, 60_000);
    ok(bash
        .run(json!({
            "command": format!("echo $$ > {}; sleep 90", pidfile.display())
        }))
        .await);
    ok(bash.run(json!({ "command": "sleep 91 & echo bg" })).await);
    ok(bash
        .run(json!({ "command": "sleep 0.2; echo quick" }))
        .await);
    notices_within(&bash, Duration::from_secs(10))
        .await
        .expect("job 3 ends");
    let pid = read_pid(&pidfile).await;
    let stopped = bash.jobs().end_turn();
    let summary: Vec<(u64, JobKind)> = stopped.iter().map(|s| (s.job, s.kind)).collect();
    assert_eq!(
        summary,
        vec![(1, JobKind::Yielded)],
        "only the running yielded job"
    );
    assert!(wait_dead(pid).await, "and it is really stopped");
    let carry = bash.jobs().carry().unwrap();
    assert!(
        carry.contains("- job 2:"),
        "the adopted job lives on: {carry}"
    );
    assert!(
        carry.contains("- job 3:"),
        "an ended job keeps its unread output: {carry}"
    );
    assert!(
        jobs.run(json!({ "job": 1 })).await.is_err(),
        "job 1 is gone"
    );
    bash.jobs().stop_all();
}

#[tokio::test]
async fn the_carry_lists_running_and_unread_jobs() {
    let (bash, _jobs) = tools(200, 60_000, 60_000);
    ok(bash.run(json!({ "command": "sleep 75" })).await);
    ok(bash.run(json!({ "command": "sleep 0.5; echo x" })).await);
    notices_within(&bash, Duration::from_secs(10))
        .await
        .expect("job 2 ends");
    let carry = bash.jobs().carry().expect("jobs in flight");
    assert!(carry.contains("do not start these again"), "{carry}");
    assert!(carry.contains("- job 1: `sleep 75` (running"), "{carry}");
    assert!(
        carry.contains("- job 2: `sleep 0.5; echo x` (ended, output not yet read"),
        "{carry}"
    );
    bash.jobs().stop_all();
}

#[tokio::test]
async fn eviction_takes_the_oldest_yielded_job_never_an_adopted_one() {
    let (bash, jobs) = tools(200, 60_000, 60_000);
    let adopted = ok(bash.run(json!({ "command": "sleep 76 & echo bg" })).await);
    assert!(adopted.contains("tracked as job 1"), "{adopted}");
    let mut last = String::new();
    for _ in 0..=MAX_YIELDED_JOBS {
        last = ok(bash.run(json!({ "command": "sleep 77" })).await);
    }
    // Jobs 2..=6 were yielded; the fifth parked evicts job 2, the oldest yielded, not job 1.
    assert!(last.contains("Job 2 was stopped"), "{last}");
    assert!(
        jobs.run(json!({ "job": 2 })).await.is_err(),
        "job 2 is gone"
    );
    let carry = bash.jobs().carry().unwrap();
    assert!(carry.contains("- job 1:"), "the adopted job stays: {carry}");
    bash.jobs().stop_all();
}

#[tokio::test]
async fn adopting_past_the_intended_cap_is_refused_with_the_list() {
    let (bash, _jobs) = tools(30_000, 60_000, 60_000);
    for i in 0..MAX_INTENDED_JOBS {
        let out = ok(bash
            .run(json!({ "command": format!("sleep 78 & echo {i}") }))
            .await);
        assert!(out.contains("tracked as job"), "{out}");
    }
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let refused = ok(bash
        .run(json!({
            "command": format!("sh -c 'echo $$ > {}; exec sleep 79' & echo one-more", pidfile.display())
        }))
        .await);
    assert!(refused.contains("It was stopped"), "{refused}");
    assert!(
        refused.contains("job 1 `sleep 78 & echo 0`"),
        "lists the running jobs: {refused}"
    );
    let pid = read_pid(&pidfile).await;
    assert!(wait_dead(pid).await, "the refused process is stopped");
    bash.jobs().stop_all();
}

#[tokio::test]
async fn a_yielded_job_whose_shell_exits_leaving_a_process_is_noticed_as_adopted() {
    let (bash, jobs) = tools(300, 60_000, 60_000);
    let out = ok(bash
        .run(json!({ "command": "sleep 0.8; sleep 80 & echo bg" }))
        .await);
    assert!(
        out.contains("No output for") && out.contains("job 1"),
        "{out}"
    );
    let notice = notices_within(&bash, Duration::from_secs(10))
        .await
        .expect("the shell's exit is news");
    assert!(
        notice.starts_with("[job 1's command exited with code 0 after"),
        "{notice}"
    );
    assert!(
        notice.contains("left a process (`sleep 80`) running"),
        "{notice}"
    );
    // Now intended: listed as left running, and never evicted.
    let carry = bash.jobs().carry().unwrap();
    assert!(carry.contains("left running by its command"), "{carry}");
    ok(jobs.run(json!({ "job": 1, "kill": true })).await);
}
