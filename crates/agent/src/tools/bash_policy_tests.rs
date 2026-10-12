//! `BashPolicy`: the detached-job path (idle and wall-clock yields, wait, kill, cleanup) and the
//! timeout-guidance text. The default policy's behavior is covered by `bash.rs`'s own tests.

use super::*;
use crate::tools::exec::ExecResult;

fn idle(ms: u64) -> BashPolicy {
    BashPolicy {
        idle_yield: Some(Duration::from_millis(ms)),
        ..BashPolicy::default()
    }
}

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

/// A detaching `bash` and the `bash_job` over it, as the registry pairs them.
fn detaching(policy: BashPolicy) -> (Arc<Bash>, BashJob) {
    let bash = Arc::new(Bash::real().with_policy(policy));
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
fn the_default_policy_leaves_the_schema_and_description_alone() {
    let bash = Bash::real();
    let schema = bash.input_schema();
    assert_eq!(schema["required"], json!(["command"]));
    assert!(schema["properties"].get("job").is_none());
    assert!(!bash.description().contains("job"));
    assert!(!bash.description().contains("Set `timeout_ms` to fit"));
    // Opting into an all-default policy is the same as not opting in.
    let same = Bash::real().with_policy(BashPolicy::default());
    assert_eq!(same.description(), bash.description());
}

#[test]
fn a_detaching_policy_keeps_the_bash_schema_and_adds_bash_job() {
    let (bash, jobs) = detaching(idle(60_000));
    assert_eq!(bash.input_schema(), Bash::real().input_schema());
    let desc = bash.description();
    assert!(desc.contains("60 s without new output"), "{desc}");
    assert!(desc.contains("`bash_job`"), "{desc}");

    let names = |reg: &agent_core::ToolRegistry| {
        reg.definitions()
            .iter()
            .map(|d| d.name.clone())
            .collect::<Vec<_>>()
    };
    let plain = crate::tools::default_registry_with_config(&crate::tools::ToolConfig::new());
    assert!(!names(&plain).contains(&"bash_job".to_string()));
    let detaching = crate::tools::default_registry_with_config(&crate::tools::ToolConfig {
        bash_policy: idle(60_000),
        ..crate::tools::ToolConfig::new()
    });
    assert!(names(&detaching).contains(&"bash_job".to_string()));
    assert_eq!(jobs.input_schema()["required"], json!(["job"]));
}

#[tokio::test]
async fn a_model_that_fills_every_field_still_runs_its_command() {
    // gpt-6-luna sends every schema property on every call; `bash` must not read a stray `job`.
    let (bash, _jobs) = detaching(idle(60_000));
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
    let (bash, jobs) = detaching(idle(500));
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
    let (bash, _jobs) = detaching(idle(1_500));
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
    let (bash, jobs) = detaching(idle(1_000));
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
async fn the_wall_yield_fires_even_while_output_flows() {
    let (bash, jobs) = detaching(BashPolicy {
        wall_yield: Some(Duration::from_millis(800)),
        ..BashPolicy::default()
    });
    let out = text(
        bash.run(json!({ "command": "while true; do echo spin; sleep 0.1; done" }))
            .await,
    );
    assert!(out.contains("spin"), "{out}");
    assert!(out.contains("Waited"), "{out}");
    assert!(out.contains("job 1"), "{out}");
    let killed = text(jobs.run(json!({ "job": 1, "kill": true })).await);
    assert!(killed.contains("Job 1 killed"), "{killed}");
}

#[tokio::test]
async fn dropping_the_tool_kills_a_parked_job() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (bash, jobs) = detaching(idle(300));
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
    let (bash, jobs) = detaching(idle(200));
    let mut last = String::new();
    for _ in 0..=MAX_PARKED_JOBS {
        last = text(bash.run(json!({ "command": "sleep 60" })).await);
    }
    let newest = MAX_PARKED_JOBS + 1;
    assert!(last.contains(&format!("job {newest}")), "{last}");
    assert!(last.contains("Job 1 was killed"), "{last}");
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
    let (bash, _jobs) = detaching(idle(30_000));
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

#[test]
fn timeout_guidance_adds_sizing_advice_to_the_description() {
    let desc = Bash::real()
        .with_policy(BashPolicy {
            timeout_guidance: true,
            ..BashPolicy::default()
        })
        .description()
        .to_string();
    assert!(
        desc.contains("Set `timeout_ms` to fit the command"),
        "{desc}"
    );
    // The default and the ceiling are unchanged; only the advice is new.
    assert!(desc.contains("30 minutes"), "{desc}");
}

#[tokio::test]
async fn timeout_guidance_adds_the_stuck_hint_only_to_a_timeout() {
    let guided = |result: ExecResult| {
        Bash::with_runner(Arc::new(FixedRunner(result))).with_policy(BashPolicy {
            timeout_guidance: true,
            ..BashPolicy::default()
        })
    };
    let ToolError::Execution(msg) = guided(ExecResult {
        timed_out: true,
        ..Default::default()
    })
    .run(json!({ "command": "x", "timeout_ms": 5000 }))
    .await
    .unwrap_err() else {
        panic!("expected an execution error")
    };
    assert!(msg.contains("Command timed out after 5 seconds"), "{msg}");
    assert!(msg.contains("faulthandler_timeout"), "{msg}");

    let ToolError::Execution(msg) = guided(ExecResult {
        code: Some(1),
        ..Default::default()
    })
    .run(json!({ "command": "x" }))
    .await
    .unwrap_err() else {
        panic!("expected an execution error")
    };
    assert!(!msg.contains("faulthandler_timeout"), "{msg}");
}

/// Returns one canned result without running anything.
struct FixedRunner(ExecResult);

#[async_trait]
impl CommandRunner for FixedRunner {
    async fn run(
        &self,
        _program: &str,
        _args: &[String],
        _cwd: Option<&str>,
        _timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        Ok(self.0.clone())
    }
}
