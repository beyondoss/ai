//! The baseline's job notify (v1.4, `exec_endpoint::jobs::ShellJobs`): the job's `setsid`
//! supervisor sends the proof and every state change with the one pinned `curl` invocation, from
//! a 0600 config the token reached on standard input, and only when the endpoint's deployment
//! advertises notify, `curl` is there and standard input gets through.
//!
//! Each test runs on every rung this host has:
//!
//! - **debian**: this host's `sh` (dash) and tools;
//! - **busybox**: `busybox sh` and busybox's applets alone on `PATH` (plus `curl`), the shell and
//!   tools of Alpine and most sandboxes;
//! - **live**: the target `AGENT_LIVE_EXEC_STDIN_CMD` names (a transport that forwards standard
//!   input, e.g. `docker exec -i <container> {}`), which CI's `exec-live` shard points at an Alpine
//!   container on the host network with `curl` added.
//!
//! When the scripts retry and when they stop is the shared kernel's (`exec_notify::SH`, held to the
//! Rust kernel by `exec_notify`'s differential test); these check the wiring around it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use beyond_agent::exec_endpoint::jobs::{
    JobId, JobNotify, JobScope, JobSpec, JobStatus, SandboxJobs, ShellJobs,
};
use beyond_agent::exec_endpoint::{HttpExecRunner, TemplateRunner};
use beyond_agent::tools::exec::{CommandRunner, ExecResult, RealRunner};
use common::exec_mock::{Answer, CallbackReceiver, ExecMock, MockOptions, exec_notify};

/// Runs every command with `PATH` set to one directory, through `env`.
struct PathRunner {
    path: String,
    /// Drop standard input, as a transport that does not forward it does.
    no_stdin: bool,
}

impl PathRunner {
    fn argv(&self, program: &str, args: &[String]) -> Vec<String> {
        let mut v = vec![format!("PATH={}", self.path), program.to_string()];
        v.extend(args.iter().cloned());
        v
    }
}

#[async_trait]
impl CommandRunner for PathRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        RealRunner
            .run("env", &self.argv(program, args), cwd, timeout)
            .await
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        if self.no_stdin {
            return self.run(program, args, cwd, timeout).await;
        }
        RealRunner
            .run_with_stdin("env", &self.argv(program, args), cwd, timeout, stdin)
            .await
    }
}

/// A directory of links: busybox's applets (when `busybox`), or the named host tools.
fn tool_dir(busybox: bool, with_curl: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let which = |t: &str| {
        let out = std::process::Command::new("sh")
            .args(["-c", &format!("command -v {t}")])
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let tools: Vec<String> = if busybox {
        let out = std::process::Command::new("busybox")
            .arg("--list")
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::to_string)
            .collect()
    } else {
        [
            "sh", "setsid", "mkfifo", "tee", "dd", "wc", "base64", "date", "sed", "od", "tr",
            "cat", "rm", "mv", "mkdir", "sleep", "chmod", "kill", "ls", "env", "grep", "stat",
        ]
        .map(str::to_string)
        .to_vec()
    };
    let busybox_path = which("busybox");
    for t in tools {
        let target = if busybox {
            busybox_path.clone()
        } else {
            which(&t)
        };
        if !target.is_empty() {
            let _ = std::os::unix::fs::symlink(target, dir.path().join(&t));
        }
    }
    if with_curl {
        std::os::unix::fs::symlink(which("curl"), dir.path().join("curl")).unwrap();
    }
    dir
}

fn has_busybox() -> bool {
    std::process::Command::new("busybox")
        .arg("true")
        .status()
        .is_ok_and(|s| s.success())
}

/// One place jobs run, and how its scripts are reached.
struct Rung {
    name: &'static str,
    runner: Arc<dyn CommandRunner>,
    root: String,
    _dirs: Vec<tempfile::TempDir>,
}

impl Rung {
    fn jobs(&self) -> ShellJobs {
        ShellJobs::new(self.runner.clone()).with_notify_advertised(true)
    }

    fn scope(&self) -> JobScope {
        JobScope::under(self.root.clone(), JobId::fresh().as_str())
    }

    async fn sh(&self, script: &str) -> ExecResult {
        self.runner
            .run(
                "sh",
                &["-c".to_string(), script.to_string()],
                None,
                Duration::from_secs(30),
            )
            .await
            .unwrap()
    }
}

/// Every rung this host has.
fn rungs() -> Vec<Rung> {
    let mut out = Vec::new();
    let root = tempfile::tempdir().unwrap();
    out.push(Rung {
        name: "debian",
        runner: Arc::new(RealRunner),
        root: root.path().to_string_lossy().into_owned(),
        _dirs: vec![root],
    });
    if has_busybox() {
        let tools = tool_dir(true, true);
        let root = tempfile::tempdir().unwrap();
        out.push(Rung {
            name: "busybox",
            runner: Arc::new(PathRunner {
                path: tools.path().to_string_lossy().into_owned(),
                no_stdin: false,
            }),
            root: root.path().to_string_lossy().into_owned(),
            _dirs: vec![tools, root],
        });
    } else {
        eprintln!("busybox is not installed here: its rung is skipped");
    }
    if let Some(cmd) = std::env::var("AGENT_LIVE_EXEC_STDIN_CMD")
        .ok()
        .filter(|c| !c.is_empty())
    {
        out.push(Rung {
            name: "live",
            runner: Arc::new(TemplateRunner::parse(&cmd).unwrap()),
            root: "/tmp/beyond-jobs".to_string(),
            _dirs: Vec::new(),
        });
    }
    out
}

fn spec(script: &str) -> JobSpec {
    JobSpec {
        program: "sh".into(),
        args: vec!["-c".into(), script.into()],
        cwd: None,
        label: script.into(),
    }
}

fn token() -> String {
    format!("bjn_v1.{}.tok-_~+/==", JobId::fresh())
}

#[tokio::test]
async fn baseline_offers_notify_only_with_curl() {
    for rung in rungs() {
        let name = rung.name;
        assert!(rung.jobs().available().await, "{name}: no jobs at all");
        assert!(
            rung.jobs().offers_notify().await,
            "{name}: advertised, with curl and stdin, it must offer notify"
        );
        assert!(
            !ShellJobs::new(rung.runner.clone()).offers_notify().await,
            "{name}: nothing advertised it"
        );
        assert!(
            !ShellJobs::new(rung.runner.clone())
                .with_notify_advertised(false)
                .offers_notify()
                .await,
            "{name}: the deployment's setting is off"
        );
    }
    // Without curl: busybox alone, and this host's tools without it.
    let mut no_curl = vec![tool_dir(false, false)];
    if has_busybox() {
        no_curl.push(tool_dir(true, false));
    }
    for dir in &no_curl {
        let jobs = ShellJobs::new(Arc::new(PathRunner {
            path: dir.path().to_string_lossy().into_owned(),
            no_stdin: false,
        }))
        .with_notify_advertised(true);
        assert!(jobs.available().await, "the jobs themselves need no curl");
        assert!(!jobs.offers_notify().await, "no curl, no notify");
        let scope = JobScope::under(dir.path().to_string_lossy(), "s");
        let id = JobId::fresh();
        jobs.start_kept(&scope, &id, &spec("sleep 5"))
            .await
            .unwrap();
        let err = jobs
            .keep_notify(
                &scope,
                &id,
                1,
                "background",
                0,
                &JobNotify {
                    url: "http://127.0.0.1:1/x".into(),
                    token: token(),
                },
            )
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported, "{err}");
        let _ = jobs.wipe(&scope).await;
    }
    // Standard input that never arrives.
    let tools = tool_dir(false, true);
    let jobs = ShellJobs::new(Arc::new(PathRunner {
        path: tools.path().to_string_lossy().into_owned(),
        no_stdin: true,
    }))
    .with_notify_advertised(true);
    assert!(jobs.available().await);
    assert!(!jobs.offers_notify().await, "no stdin, no notify");

    // Over an exec endpoint, its probe answer is the setting.
    for (setting, want) in [(true, true), (false, false)] {
        let dir = tempfile::tempdir().unwrap();
        let mock = ExecMock::start_with(
            dir.path(),
            MockOptions {
                jobs: false,
                kept_jobs_survive_idle: setting,
                ..MockOptions::default()
            },
        )
        .await;
        let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
        let jobs = runner.sandbox_jobs().unwrap();
        assert_eq!(jobs.offers_notify().await, want, "setting {setting}");
    }
}

#[tokio::test]
async fn the_supervisor_sends_the_proof() {
    for rung in rungs() {
        let name = rung.name;
        let jobs = rung.jobs();
        let (scope, id) = (rung.scope(), JobId::fresh());
        let rx = CallbackReceiver::start().await;
        // The proof's answer is slow, so its curl can be seen at work.
        rx.script([Answer::Slow(Duration::from_secs(5), 200)]);
        jobs.start_kept(&scope, &id, &spec("sleep 7; sleep 2 & exit 0"))
            .await
            .unwrap();
        let tok = token();
        let t = Instant::now();
        jobs.keep_notify(
            &scope,
            &id,
            4,
            "background",
            0,
            &JobNotify {
                url: rx.url.clone(),
                token: tok.clone(),
            },
        )
        .await
        .unwrap();
        let answered = t.elapsed();
        assert!(answered < Duration::from_secs(5), "{name}: keep waited");
        let got = rx.wait_for(1, Duration::from_secs(5)).await;
        assert_eq!(got.len(), 1, "{name}: no proof");
        let proof = &got[0];
        assert_eq!(
            proof.body,
            exec_notify::body(scope.name(), id.as_str()).as_bytes(),
            "{name}"
        );
        assert_eq!(
            proof.header_values("authorization"),
            vec![format!("Bearer {tok}").as_str()],
            "{name}"
        );
        assert_eq!(
            proof.header_values("content-type"),
            vec!["application/json"],
            "{name}"
        );
        assert_eq!(proof.method, "POST");
        // The attempt in flight: exactly the pinned invocation, run outside the keep's group.
        let dir = format!("{}/{}/{}", scope.root(), scope.name(), id);
        let ps = rung
            .sh(&format!(
                "for f in /proc/[0-9]*/cmdline; do c=$(tr '\\000' ' ' <\"$f\" 2>/dev/null) || continue; \
                 case $c in *\"{dir}/notify\"*) echo \"$c\";; esac; done"
            ))
            .await;
        let want = format!(
            "curl -q -sS -K {dir}/notify -o /dev/null -w %{{http_code}} --max-time 30 --proto =http,https "
        );
        assert!(
            ps.stdout.lines().any(|l| l == want),
            "{name}: want `{want}` in flight, saw {:?}",
            ps.stdout
        );
        let mode = rung.sh(&format!("stat -c %a {dir}/notify")).await;
        assert_eq!(mode.stdout.trim(), "600", "{name}");
        assert!(
            rung.sh(&format!("test -e {dir}/notifier")).await.code == Some(0),
            "{name}: the supervisor records that it carries the notifier"
        );
        // The shell's exit with its leftover alive, and the end.
        let got = rx.wait_for(3, Duration::from_secs(16)).await;
        assert!(
            got.len() >= 3,
            "{name}: want the proof, the shell's exit and the end; got {}",
            got.len()
        );
        let listed = jobs.list(&scope).await.unwrap();
        assert_eq!(listed[0].status, JobStatus::Ended(0), "{name}");
        let _ = jobs.wipe(&scope).await;
    }
}

#[tokio::test]
async fn an_older_supervisor_refuses_notify() {
    for rung in rungs() {
        let name = rung.name;
        let jobs = rung.jobs();
        let scope = rung.scope();
        let notify = JobNotify {
            url: "http://127.0.0.1:9/_beyond/agent?session_id=s".into(),
            token: token(),
        };
        // Started by a supervisor without the notifier: its directory has no record of one.
        let id = JobId::fresh();
        jobs.start_kept(&scope, &id, &spec("sleep 30"))
            .await
            .unwrap();
        let dir = format!("{}/{}/{}", scope.root(), scope.name(), id);
        rung.sh(&format!("rm -f {dir}/notifier")).await;
        let err = jobs
            .keep_notify(&scope, &id, 2, "background", 0, &notify)
            .await
            .unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidInput,
            "{name}: {err}"
        );
        let job = jobs.list(&scope).await.unwrap().remove(0);
        assert_eq!(job.num, None, "{name}: a refused keep recorded nothing");
        assert!(
            rung.sh(&format!("test -e {dir}/notify")).await.code != Some(0),
            "{name}: no notify stored"
        );
        // A job that ended before any notify: nothing is left to send it.
        let ended = JobId::fresh();
        jobs.start_kept(&scope, &ended, &spec("true"))
            .await
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !jobs
            .list(&scope)
            .await
            .unwrap()
            .iter()
            .any(|j| j.id == ended && matches!(j.status, JobStatus::Ended(_)))
        {
            assert!(Instant::now() < deadline, "{name}: never ended");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let err = jobs
            .keep_notify(&scope, &ended, 3, "background", 0, &notify)
            .await
            .unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidInput,
            "{name}: {err}"
        );
        // A notify that fails the grammar never reaches the sandbox.
        let bad = JobNotify {
            url: "http://edge.example/x".into(),
            token: token(),
        };
        let err = jobs
            .keep_notify(&scope, &id, 2, "background", 0, &bad)
            .await
            .unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidInput,
            "{name}: {err}"
        );
        let _ = jobs.wipe(&scope).await;
    }
}

#[tokio::test]
async fn the_token_is_never_logged() {
    for rung in rungs() {
        let name = rung.name;
        let jobs = rung.jobs();
        let (scope, id) = (rung.scope(), JobId::fresh());
        let rx = CallbackReceiver::start().await;
        rx.script([Answer::Status(503)]);
        jobs.start_kept(&scope, &id, &spec("echo out; sleep 2"))
            .await
            .unwrap();
        let tok = token();
        let notify = JobNotify {
            url: rx.url.clone(),
            token: tok.clone(),
        };
        assert!(
            !format!("{notify:?}").contains(&tok),
            "Debug shows the token"
        );
        jobs.keep_notify(&scope, &id, 1, "background", 0, &notify)
            .await
            .unwrap();
        // The proof, refused, then its retry, which (the job having ended in its wait) carries
        // the end too.
        let got = rx.wait_for(2, Duration::from_secs(10)).await.len();
        assert!(got >= 2, "{name}: {got} callbacks");
        let dir = format!("{}/{}/{}", scope.root(), scope.name(), id);
        // Every file the job has, but its 0600 config, and every process's arguments and
        // environment: the token is in none. The scan carries it in two halves.
        let (a, b) = tok.split_at(tok.len() / 2);
        let script = "t=\"$1$2\"; d=$3; \
                      for f in \"$d\"/* /proc/[0-9]*/cmdline /proc/[0-9]*/environ; do \
                      [ \"$f\" = \"$d/notify\" ] && continue; \
                      c=$(tr '\\000' '\\n' <\"$f\" 2>/dev/null) || continue; \
                      case $c in *\"$t\"*) echo \"$f\";; esac; done";
        let found = rung
            .runner
            .run(
                "sh",
                &[
                    "-c".into(),
                    script.into(),
                    "sh".into(),
                    a.into(),
                    b.into(),
                    dir.clone(),
                ],
                None,
                Duration::from_secs(30),
            )
            .await
            .unwrap();
        assert_eq!(found.stdout.trim(), "", "{name}: the token is in these");
        let read = jobs.read(&scope, &id, 0, 4096).await.unwrap();
        assert!(
            !String::from_utf8_lossy(&read.data).contains(&tok),
            "{name}"
        );
        assert!(
            !format!("{:?}", jobs.list(&scope).await.unwrap()).contains(&tok),
            "{name}"
        );
        let _ = jobs.wipe(&scope).await;
    }
}

/// A notifier that dies is started again: the watcher removes the lock of a notifier whose pid is
/// gone (here killed with `SIGKILL`, which runs no trap), a new one takes it, and, being a fresh
/// kernel, owes the latest keep's callback anew. A notifier that finishes removes its own lock.
#[tokio::test]
async fn a_notifier_that_dies_is_started_again() {
    for rung in rungs() {
        let name = rung.name;
        let jobs = rung.jobs();
        let (scope, id) = (rung.scope(), JobId::fresh());
        let rx = CallbackReceiver::start().await;
        jobs.start_kept(&scope, &id, &spec("sleep 6"))
            .await
            .unwrap();
        jobs.keep_notify(
            &scope,
            &id,
            1,
            "background",
            0,
            &JobNotify {
                url: rx.url.clone(),
                token: token(),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            rx.wait_for(1, Duration::from_secs(5)).await.len(),
            1,
            "{name}: the proof"
        );
        let dir = format!("{}/{}/{}", scope.root(), scope.name(), id);
        let pid_cmd = format!("cat {dir}/nlock/pid 2>/dev/null");
        let first = rung.sh(&pid_cmd).await.stdout.trim().to_string();
        assert!(!first.is_empty(), "{name}: the notifier records its pid");
        rung.sh(&format!("kill -KILL {first}")).await;
        // Started again, with a pid of its own, and it calls back again.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let now = rung.sh(&pid_cmd).await.stdout.trim().to_string();
            if !now.is_empty() && now != first {
                break;
            }
            assert!(Instant::now() < deadline, "{name}: never started again");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(
            rx.wait_for(2, Duration::from_secs(5)).await.len() >= 2,
            "{name}: the new notifier owes the keep's callback"
        );
        // The end is called back, and the notifier, done, removes its lock.
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let gone = rung
                .sh(&format!("test -e {dir}/end && test ! -d {dir}/nlock"))
                .await;
            if gone.code == Some(0) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "{name}: the lock outlived its notifier"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let _ = jobs.wipe(&scope).await;
    }
}

/// A job kept with notify outlives its hold, whatever its notifier does: its notifier is killed
/// and started again while the hold is up, then the hold closes, and the job runs on. Under each
/// rung's shell (dash here, busybox's `sh`, and the live target's).
#[tokio::test]
async fn a_job_kept_with_notify_outlives_its_hold() {
    use beyond_agent::exec_endpoint::jobs::JobEvent;
    for rung in rungs() {
        let name = rung.name;
        let jobs = rung.jobs();
        if !jobs.holds().await {
            eprintln!("{name}: this rung cannot hold a job");
            continue;
        }
        let (scope, id) = (rung.scope(), JobId::fresh());
        let rx = CallbackReceiver::start().await;
        let started = std::sync::Mutex::new(None::<u32>);
        let events = |e: JobEvent<'_>| {
            if let JobEvent::Started(g) = e {
                *started.lock().unwrap() = Some(g);
            }
        };
        let s = spec("sleep 30");
        let mut run = Box::pin(jobs.run_held(&scope, &id, &s, &events));
        let deadline = Instant::now() + Duration::from_secs(10);
        while started.lock().unwrap().is_none() {
            tokio::select! {
                r = &mut run => panic!("{name}: ended early: {r:?}"),
                () = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
            assert!(Instant::now() < deadline, "{name}: never started");
        }
        let pgid = started.lock().unwrap().unwrap();
        // Kept with notify while the hold is up (the hold keeps being polled meanwhile).
        let notify = JobNotify {
            url: rx.url.clone(),
            token: token(),
        };
        let keep = jobs.keep_notify(&scope, &id, 1, "background", 0, &notify);
        tokio::select! {
            r = &mut run => panic!("{name}: ended early: {r:?}"),
            k = keep => k.unwrap(),
        }
        let proof = async { rx.wait_for(1, Duration::from_secs(5)).await };
        tokio::select! {
            r = &mut run => panic!("{name}: ended early: {r:?}"),
            got = proof => assert_eq!(got.len(), 1, "{name}: the proof"),
        }
        // Its notifier dies and is started again, the hold still up.
        let dir = format!("{}/{}/{}", scope.root(), scope.name(), id);
        let pid_cmd = format!("cat {dir}/nlock/pid 2>/dev/null");
        let first = rung.sh(&pid_cmd).await.stdout.trim().to_string();
        assert!(!first.is_empty(), "{name}: the notifier records its pid");
        rung.sh(&format!("kill -KILL {first}")).await;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let now = rung.sh(&pid_cmd).await.stdout.trim().to_string();
            if !now.is_empty() && now != first {
                break;
            }
            assert!(Instant::now() < deadline, "{name}: never started again");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        // The hold closes: the kept job runs on.
        drop(run);
        tokio::time::sleep(Duration::from_secs(2)).await;
        let alive = rung.sh(&format!("kill -0 -{pgid} 2>/dev/null")).await;
        assert_eq!(alive.code, Some(0), "{name}: a kept job died with its hold");
        let _ = jobs.wipe(&scope).await;
    }
}
