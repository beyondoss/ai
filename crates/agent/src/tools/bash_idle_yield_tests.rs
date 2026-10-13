//! `bash`'s idle yield: a quiet command becomes a job (wait, kill, cleanup, the growing window), and
//! the wait-for-exit mode `serve`'s host command uses. The rest of `bash` is covered by `bash.rs`'s
//! own tests.

use super::*;

fn text(r: Result<ToolOutput, ToolError>) -> String {
    match r {
        Ok(out) => out.text,
        Err(e) => panic!("expected success, got {e}"),
    }
}

/// Whether a process with this pid is still alive (signal 0 probes without signalling).
fn alive(pid: &str) -> bool {
    std::process::Command::new("kill")
        .args(["-0", pid])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// A `bash` with a `ms` idle window and the `bash_job` over it, as the registry pairs them.
fn detaching(ms: u64) -> (Arc<Bash>, BashJob) {
    let bash = Arc::new(Bash::real().with_idle_yield(Some(Duration::from_millis(ms))));
    let jobs = BashJob::new(bash.clone());
    (bash, jobs)
}

async fn wait_dead(pid: &str) -> bool {
    for _ in 0..50 {
        if !alive(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

#[test]
fn bash_yields_after_sixty_quiet_seconds_and_bash_job_rides_with_it() {
    let bash = Bash::real();
    assert!(bash.detaches());
    let desc = bash.description();
    assert!(desc.contains("60 s without new output"), "{desc}");
    assert!(desc.contains("`bash_job`"), "{desc}");
    // `bash`'s own schema has no job fields (see `input_schema`).
    let schema = bash.input_schema();
    assert_eq!(schema["required"], json!(["command"]));
    assert!(schema["properties"].get("job").is_none());

    let names = |reg: &agent_core::ToolRegistry| {
        reg.definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect::<Vec<_>>()
    };
    let has_job = |reg: &agent_core::ToolRegistry| names(reg).contains(&JOB_TOOL.to_string());
    let reg = crate::tools::default_registry_with_config(&crate::tools::ToolConfig::new());
    assert!(has_job(&reg));
    let waits = crate::tools::default_registry_with_config(&crate::tools::ToolConfig {
        bash_waits_for_exit: true,
        ..crate::tools::ToolConfig::new()
    });
    assert!(!has_job(&waits));
    // Filters keep or drop `bash_job` with `bash`, whatever they name.
    let only_bash = ["read".to_string(), "bash".to_string()];
    let mut allowed = crate::tools::default_registry();
    crate::tools::apply_filter(&mut allowed, Some(&only_bash), None, false);
    assert!(has_job(&allowed));
    let mut excluded = crate::tools::default_registry();
    crate::tools::apply_filter(&mut excluded, None, Some(&["bash".to_string()]), false);
    assert!(!has_job(&excluded));
    let (_, jobs) = detaching(60_000);
    let schema = jobs.input_schema();
    assert_eq!(schema["required"], json!(["job"]));
    // `timeout_ms` on both tools, in the same words, with this instance's defaults.
    let job_timeout = schema["properties"]["timeout_ms"]["description"]
        .as_str()
        .unwrap();
    assert!(
        job_timeout.starts_with(
            "How long to wait, in milliseconds, before returning with the output so far; 0 \
             returns at once."
        ),
        "{job_timeout}"
    );
    assert!(
        job_timeout.ends_with("after 60 s without output, or after 5 minutes."),
        "{job_timeout}"
    );
    let bash_timeout = bash.input_schema()["properties"]["timeout_ms"]["description"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        bash_timeout.starts_with(
            "How long to wait, in milliseconds, before returning with the output so far."
        ) && bash_timeout.ends_with("after 60 s without output, or after 10 minutes."),
        "{bash_timeout}"
    );
    let desc = jobs.description();
    assert!(
        desc.contains("never for a server or watcher to exit"),
        "{desc}"
    );
}

#[tokio::test]
async fn a_model_that_fills_every_field_still_runs_its_command() {
    // gpt-6-luna sends every schema property on every call; `bash` must not read a stray `job`.
    let (bash, _jobs) = detaching(60_000);
    let out = text(
        bash.run(json!({ "command": "echo ran", "job": 0, "kill": false }))
            .await,
    );
    assert!(out.contains("ran"), "{out}");
}

#[tokio::test]
async fn a_quiet_command_returns_early_and_keeps_running_until_killed() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (bash, jobs) = detaching(500);
    let started = Instant::now();
    let out = text(
        bash.run(json!({
            "command": format!("echo $$ > {}; echo before-the-hang; sleep 60", pidfile.display())
        }))
        .await,
    );
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "returned after the idle yield, not the command"
    );
    assert!(out.contains("before-the-hang"), "{out}");
    assert!(out.contains("No output for"), "{out}");
    assert!(out.contains("job 1"), "{out}");
    let pid = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .to_string();
    assert!(alive(&pid), "the command must keep running after the yield");

    let killed = text(jobs.run(json!({ "job": 1, "kill": true })).await);
    assert!(killed.contains("Job 1 killed"), "{killed}");
    assert!(wait_dead(&pid).await, "kill must stop the command");

    let gone = jobs.run(json!({ "job": 1 })).await.unwrap_err();
    assert!(matches!(gone, ToolError::InvalidInput(_)), "{gone}");
}

#[tokio::test]
async fn a_command_that_keeps_printing_is_never_interrupted() {
    let (bash, _jobs) = detaching(1_500);
    let out = text(
        bash.run(json!({
            "command": "for i in 1 2 3 4 5 6; do echo tick-$i; sleep 0.4; done"
        }))
        .await,
    );
    assert!(out.contains("tick-1") && out.contains("tick-6"), "{out}");
    assert!(
        !out.contains("job"),
        "a steady command must finish in one call: {out}"
    );
}

#[tokio::test]
async fn waiting_on_a_job_reports_only_the_new_output_and_its_exit() {
    // The re-attach starts right after the first yield (~1 s) and gets a fresh 1 s of quiet, so
    // `second` at ~1.5 s lands inside it and the job runs to its exit in that call.
    let (bash, jobs) = detaching(1_000);
    let first = text(
        bash.run(json!({ "command": "echo first; sleep 1.5; echo second; exit 3" }))
            .await,
    );
    assert!(
        first.contains("first") && first.contains("job 1"),
        "{first}"
    );
    let ToolError::Execution(done) = jobs.run(json!({ "job": "1" })).await.unwrap_err() else {
        panic!("a non-zero exit is an execution error")
    };
    assert!(done.contains("second"), "{done}");
    assert!(!done.contains("first"), "already reported: {done}");
    assert!(done.contains("Command exited with code 3"), "{done}");
    // Reported once; the job is gone.
    assert!(jobs.run(json!({ "job": 1 })).await.is_err());
}

#[tokio::test]
async fn every_wait_on_a_silent_job_gets_the_same_silence_window() {
    // No growth: with exits pushed, a wait returning early costs a turn, and a grown window cost a
    // 240 s wait on a watcher in the background A/B. (It used to be 0.5 s, then 2 s, here.)
    let (bash, jobs) = detaching(500);
    let first = text(bash.run(json!({ "command": "sleep 30" })).await);
    assert!(first.contains("No output for"), "{first}");
    assert!(
        first.contains(r#"if you need its result, wait for it with bash_job {"job": 1}."#),
        "{first}"
    );
    for _ in 0..2 {
        let started = Instant::now();
        let waited = text(jobs.run(json!({ "job": 1 })).await);
        let took = started.elapsed();
        assert!(
            took >= Duration::from_millis(500) && took < Duration::from_millis(1_500),
            "one window, not a grown one: {took:?}"
        );
        // A wait's report is short: the model was told how to use the job when it started.
        assert!(
            waited.starts_with("No output for 0 s. Job 1 is still running ("),
            "{waited}"
        );
        assert!(!waited.contains("pkill"), "{waited}");
    }
    text(jobs.run(json!({ "job": 1, "kill": true })).await);
}

#[tokio::test]
async fn waiting_for_exit_never_yields() {
    // `serve`'s host `bash` command has no `bash_job` to come back with.
    let bash = Bash::real().with_idle_yield(None);
    assert!(!bash.detaches());
    assert!(!bash.description().contains("bash_job"));
    let out = text(bash.run(json!({ "command": "sleep 0.6; echo done" })).await);
    assert!(out.contains("done") && !out.contains("job"), "{out}");
}

#[tokio::test]
async fn dropping_the_tool_kills_a_parked_job() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (bash, jobs) = detaching(300);
    let out = text(
        bash.run(json!({
            "command": format!("echo $$ > {}; sleep 60", pidfile.display())
        }))
        .await,
    );
    assert!(out.contains("job 1"), "{out}");
    let pid = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .to_string();
    assert!(alive(&pid));
    drop((bash, jobs));
    assert!(
        wait_dead(&pid).await,
        "a parked job must not outlive its tool"
    );
}

#[tokio::test]
async fn parking_past_the_cap_kills_the_oldest_job() {
    let (bash, jobs) = detaching(200);
    let mut last = String::new();
    for _ in 0..=MAX_YIELDED_JOBS {
        last = text(bash.run(json!({ "command": "sleep 60" })).await);
    }
    let newest = MAX_YIELDED_JOBS + 1;
    assert!(last.contains(&format!("job {newest}")), "{last}");
    assert!(last.contains("Job 1 was stopped"), "{last}");
    assert!(
        jobs.run(json!({ "job": 1 })).await.is_err(),
        "job 1 is gone"
    );
    for id in 2..=newest {
        let killed = text(jobs.run(json!({ "job": id, "kill": true })).await);
        assert!(killed.contains("killed"), "{killed}");
    }
}

#[tokio::test]
async fn cancelling_a_detachable_call_kills_the_command_and_keeps_its_output() {
    let (tx, rx) = futures::channel::mpsc::unbounded();
    let cancel = agent_core::CancellationToken::new();
    let progress = ToolProgress::new(tx, "id".into(), "bash".into(), cancel.clone());
    let (bash, _jobs) = detaching(30_000);
    let run = bash.run_streaming(
        json!({ "command": "printf 'partial-output\\n'; sleep 30" }),
        &progress,
    );
    tokio::pin!(run);
    tokio::select! {
        _ = &mut run => panic!("must not complete on its own within this window"),
        () = tokio::time::sleep(Duration::from_millis(300)) => {}
    }
    cancel.cancel();
    let err = tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("cancellation must resolve promptly")
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("partial-output") && msg.contains("cancelled"),
        "{msg}"
    );
    drop(rx);
}
