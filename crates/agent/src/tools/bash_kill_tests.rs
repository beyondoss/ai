//! Stopping a job: SIGTERM first, a short grace, then the SIGKILL sweep — so a `sudo`'d command's
//! root child (which we may not signal, and which `sudo` cannot be made to relay a SIGKILL to) is
//! stopped through `sudo`'s TERM relay, and a process that survives anyway is reported, never claimed
//! "killed". Linux-only, like `bash_jobs_tests`. The `sudo` cases need passwordless `sudo -n` and
//! skip without it (CI).

use super::*;

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

/// The pids whose command line is exactly `cmd` (a unique marker per test).
fn pids_of(cmd: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|s| s.parse().ok()) else {
            continue;
        };
        let raw = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
        let line = raw
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect::<Vec<_>>()
            .join(" ");
        if line == cmd && alive(pid) {
            out.push(pid);
        }
    }
    out
}

async fn wait_for(cmd: &str) -> Vec<u32> {
    for _ in 0..100 {
        let pids = pids_of(cmd);
        if !pids.is_empty() {
            return pids;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("`{cmd}` never started");
}

fn sudo_works() -> bool {
    std::process::Command::new("sudo")
        .args(["-n", "true"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Root processes a failed test leaves behind: killed with the privilege that started them.
struct SudoReap(Vec<&'static str>);

impl Drop for SudoReap {
    fn drop(&mut self) {
        for cmd in &self.0 {
            for pid in pids_of(cmd) {
                let _ = std::process::Command::new("sudo")
                    .args(["-n", "kill", "-KILL", &pid.to_string()])
                    .status();
            }
        }
    }
}

fn tools() -> (Arc<Bash>, BashJob) {
    let bash = Arc::new(Bash::real().with_idle_yield(Some(Duration::from_millis(200))));
    let job = BashJob::new(bash.clone());
    (bash, job)
}

async fn background(bash: &Bash, command: &str) {
    let out = bash
        .run(json!({ "command": command, "run_in_background": true }))
        .await
        .unwrap_or_else(|e| panic!("start failed: {e}"));
    assert!(
        out.text.contains("Running in the background as job"),
        "{}",
        out.text
    );
}

#[tokio::test]
async fn a_kill_sends_sigterm_first() {
    let dir = tempfile::tempdir().unwrap();
    let mark = dir.path().join("term");
    let (bash, jobs) = tools();
    background(
        &bash,
        &format!(
            "trap 'echo term > {}; exit 0' TERM; while :; do sleep 0.05; done",
            mark.display()
        ),
    )
    .await;
    let killed = jobs.run(json!({ "job": 1, "kill": true })).await.unwrap();
    assert!(killed.text.contains("Job 1 killed"), "{}", killed.text);
    assert_eq!(
        std::fs::read_to_string(&mark).unwrap_or_default(),
        "term\n",
        "the shell heard SIGTERM before any SIGKILL"
    );
}

#[tokio::test]
async fn a_process_that_ignores_sigterm_is_still_killed() {
    let (bash, jobs) = tools();
    background(&bash, "trap '' TERM; exec sleep 62.31").await;
    let pids = wait_for("sleep 62.31").await;
    let killed = jobs.run(json!({ "job": 1, "kill": true })).await.unwrap();
    assert!(killed.text.contains("Job 1 killed"), "{}", killed.text);
    for pid in pids {
        assert!(wait_dead(pid).await, "the sweep's SIGKILL got it");
    }
}

/// `sudo` relays the SIGTERM to its root child; a SIGKILL would have taken `sudo` down alone and
/// left the child running (and the kill claiming "killed").
#[tokio::test]
async fn killing_a_sudo_job_stops_its_root_child() {
    if !sudo_works() {
        eprintln!("skipped: no passwordless sudo");
        return;
    }
    let _reap = SudoReap(vec!["sleep 62.32"]);
    let (bash, jobs) = tools();
    background(&bash, "sudo -n sleep 62.32").await;
    let pids = wait_for("sleep 62.32").await;
    let killed = jobs.run(json!({ "job": 1, "kill": true })).await.unwrap();
    assert!(killed.text.contains("Job 1 killed"), "{}", killed.text);
    for pid in pids {
        assert!(wait_dead(pid).await, "the root child outlived the kill");
    }
}

/// A root process that ignores the TERM `sudo` relays cannot be SIGKILLed by us: the kill and the
/// stop say so, rather than "killed".
#[tokio::test]
async fn a_root_process_that_survives_is_reported_not_claimed_killed() {
    if !sudo_works() {
        eprintln!("skipped: no passwordless sudo");
        return;
    }
    let _reap = SudoReap(vec!["sleep 62.33", "sleep 62.34"]);
    let (bash, jobs) = tools();
    background(&bash, "sudo -n sh -c \"trap '' TERM; exec sleep 62.33\"").await;
    wait_for("sleep 62.33").await;
    let killed = jobs.run(json!({ "job": 1, "kill": true })).await;
    let text = match killed {
        Err(ToolError::Execution(text)) => text,
        other => panic!("expected a failed kill, got {other:?}"),
    };
    assert!(!text.contains("killed after"), "{text}");
    assert!(
        text.contains("`sleep 62.33`") && text.contains("could not be killed"),
        "{text}"
    );

    background(&bash, "sudo -n sh -c \"trap '' TERM; exec sleep 62.34\"").await;
    wait_for("sleep 62.34").await;
    let stopped = bash.jobs().stop_all();
    assert_eq!(stopped.len(), 1, "{stopped:?}");
    let survived = stopped[0].survived.as_deref().unwrap_or_default();
    assert!(survived.contains("`sleep 62.34`"), "{stopped:?}");
}
