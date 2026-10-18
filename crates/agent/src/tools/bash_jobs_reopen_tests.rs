//! A reopened session's jobs (`Background::resume`): no job survives the process or the switch that
//! closed its session, so a fresh table numbers new jobs past every id the transcript used and tells
//! the first run which jobs the transcript left running are gone. Before, ids restarted at 1 — the
//! model took a dead server for live, and `bash_job {job: 1, kill}` killed a new, unrelated job.
//! Linux-only, like `bash_jobs_tests`. The `serve` restart and `run --continue` paths are
//! `tests/bash_jobs_reopen.rs`.

use super::*;
use agent_core::{Background, ContentBlock, Message};

fn ok(r: Result<ToolOutput, ToolError>) -> String {
    match r {
        Ok(out) => out.text,
        Err(e) => panic!("expected success, got {e}"),
    }
}

fn call(id: &str, name: &str, input: Value) -> Message {
    Message::assistant(vec![ContentBlock::tool_use(id, name, input)])
}

fn result_with(id: &str, content: &str, notes: &[&str]) -> Message {
    let mut blocks = vec![ContentBlock::ToolResult {
        tool_use_id: id.into(),
        content: content.into(),
        is_error: false,
        images: Vec::new(),
    }];
    blocks.extend(notes.iter().map(|n| ContentBlock::text(*n)));
    Message::tool_results(blocks)
}

/// A transcript in which job 1 (a server) was left running, job 2 ended (noticed), job 3 was
/// killed, job 4 was a yielded command its turn stopped, and job 5 was adopted from a yield.
fn transcript() -> Vec<Message> {
    vec![
        Message::user("start the dev servers"),
        call(
            "a",
            "bash",
            json!({ "command": "npm run dev", "run_in_background": true }),
        ),
        result_with(
            "a",
            "ready\n\nRunning in the background as job 1 (it keeps running after your turn ends).",
            &[],
        ),
        call(
            "b",
            "bash",
            json!({ "command": "cargo build", "run_in_background": true }),
        ),
        result_with(
            "b",
            "Running in the background as job 2 (it keeps running after your turn ends).",
            &[],
        ),
        call(
            "c",
            "bash",
            json!({ "command": "tail -f log", "run_in_background": true }),
        ),
        result_with(
            "c",
            "Running in the background as job 3 (it keeps running after your turn ends).",
            &["[job 2 exited with code 0 after 31 s: `cargo build`]\nok"],
        ),
        call("d", "bash_job", json!({ "job": 3, "kill": true })),
        result_with("d", "Job 3 (`tail -f log`) killed after 4 s.", &[]),
        call("e", "bash", json!({ "command": "make test" })),
        result_with(
            "e",
            "No output for 10 s. The command is still running as job 4 (4 s so far; it is stopped \
             when you end your turn).",
            &[],
        ),
        call("f", "bash", json!({ "command": "./serve.sh" })),
        result_with(
            "f",
            "No output for 10 s. The command is still running as job 5 (10 s so far).",
            &[],
        ),
        Message::assistant(vec![ContentBlock::text("Both are up.")]),
        Message::user(
            "[job 5's command exited with code 0 after 12 s but left a process (`node`) running: \
             `./serve.sh`] That is still job 5.",
        ),
        Message::assistant(vec![ContentBlock::text("Noted.")]),
        // The reopened run's own prompt.
        Message::user("is the server still up?"),
    ]
}

#[test]
fn a_reopened_transcript_lists_what_it_left_running_as_stopped() {
    let jobs = BashJobs::default();
    let note = jobs.resume(&transcript()).expect("jobs were left running");
    assert!(
        note.starts_with(agent_core::background::REMINDER_PREFIX),
        "a harness note: {note}"
    );
    assert!(
        note.contains("- job 1: `npm run dev` (not running)"),
        "{note}"
    );
    assert!(
        note.contains("- job 5: `./serve.sh` (not running)"),
        "{note}"
    );
    for gone in ["job 2", "job 3", "job 4"] {
        assert!(!note.contains(gone), "{gone} had already ended: {note}");
    }
    assert!(note.contains("New jobs are numbered from 6."), "{note}");
    // Once per transcript: the next run on it is not told again.
    assert_eq!(jobs.resume(&transcript()), None);
}

/// A reopen whose sandbox has not been read back yet (`bash_sandbox::reattach_later`) does not
/// guess: the jobs the sandbox may still hold are "not yet checked". Once the read-back settles,
/// one it did not find is reported gone, once, by the next notices.
#[test]
fn a_reopen_not_yet_read_back_calls_its_jobs_unchecked_and_reports_the_gone_ones_later() {
    let jobs = Arc::new(BashJobs::default());
    super::super::bash_sandbox::reattach_later(&jobs, Arc::new(super::super::exec::RealRunner));
    assert!(jobs.reattach_pending());
    let note = jobs.resume(&transcript()).expect("jobs were left running");
    assert!(
        note.contains("- job 1: `npm run dev` (not yet checked)"),
        "{note}"
    );
    assert!(
        note.contains("- job 5: `./serve.sh` (not yet checked)"),
        "{note}"
    );
    assert!(
        !note.contains("(not running)") && !note.contains("(still running)"),
        "nothing is claimed that nobody checked: {note}"
    );
    assert_eq!(jobs.take_notices(), None, "nothing is known yet");

    // The read-back ran and found neither.
    jobs.set_pending_reattach(None);
    let notices = jobs.take_notices().expect("the gone jobs are reported");
    assert!(
        notices.contains("[job 1 is no longer running: `npm run dev`]"),
        "{notices}"
    );
    assert!(notices.contains("[job 5 is no longer running"), "{notices}");
    assert_eq!(jobs.take_notices(), None, "once");
}

#[test]
fn a_transcript_with_no_job_left_running_gets_no_note_but_still_seeds_the_ids() {
    let jobs = BashJobs::default();
    let msgs = vec![
        Message::user("build"),
        call(
            "a",
            "bash",
            json!({ "command": "make", "run_in_background": true }),
        ),
        result_with(
            "a",
            "Running in the background as job 7 (it keeps running after your turn ends).",
            &["[job 7 exited with code 0 after 3 s: `make`]"],
        ),
        Message::assistant(vec![ContentBlock::text("done")]),
        Message::user("again"),
    ];
    assert_eq!(jobs.resume(&msgs), None);
    assert_eq!(jobs.last_id(), 7);
}

#[test]
fn ids_are_read_from_a_summary_and_from_other_spellings() {
    let jobs = BashJobs::default();
    let msgs = vec![
        Message::user(
            "## Goal\nThe dev server ran as Job 12; bash_job {\"job\": 9} waited on a build.",
        ),
        Message::user("next"),
    ];
    assert_eq!(jobs.resume(&msgs), None, "nothing is known to be running");
    assert_eq!(jobs.last_id(), 12);
}

#[test]
fn a_transcript_switch_reads_the_new_transcript() {
    let jobs = BashJobs::default();
    assert!(jobs.resume(&transcript()).is_some());
    // `serve` stops every job on a switch, which resets the table for the transcript it switched to.
    jobs.stop_all();
    let note = jobs
        .resume(&transcript())
        .expect("read again after a switch");
    assert!(
        note.contains("- job 1: `npm run dev` (not running)"),
        "{note}"
    );
}

#[tokio::test]
async fn a_new_job_after_a_reopen_takes_a_fresh_id_and_a_kill_names_its_command() {
    let bash = Arc::new(
        Bash::real()
            .with_idle_yield(Some(Duration::from_millis(300)))
            .with_settle(Duration::from_millis(200)),
    );
    let job_tool = BashJob::new(bash.clone());
    assert!(bash.jobs().resume(&transcript()).is_some());
    let out = ok(bash
        .run(json!({ "command": "sleep 60", "run_in_background": true }))
        .await);
    assert!(out.contains("as job 6 "), "never reuses job 1: {out}");
    // The old id names nothing now, rather than the new job.
    assert!(
        job_tool
            .run(json!({ "job": 1, "kill": true }))
            .await
            .is_err()
    );
    let killed = ok(job_tool.run(json!({ "job": 6, "kill": true })).await);
    assert!(
        killed.contains("Job 6 (`sleep 60`) killed after"),
        "{killed}"
    );
}

/// A process carrying a job tag, as a killed agent's job leaves one: the reopen finds it by its tag.
fn tagged_sleep(tag: &str) -> std::process::Child {
    std::process::Command::new("sleep")
        .arg("60")
        .env(super::super::bash_proc::TAG_ENV, tag)
        .spawn()
        .unwrap()
}

#[test]
fn a_job_a_killed_process_left_running_is_stopped_on_reopen_and_reported_so() {
    let jobs = BashJobs::default();
    let tag = format!("reopen-test.{}", std::process::id());
    let mut survivor = tagged_sleep(&tag);
    let pid = survivor.id();
    let fates = jobs.stop_survivors(&[
        // As journaled at spawn, then again with its job number once parked.
        json!({ "tag": tag, "pgid": pid, "start": null, "command": "npm run dev" }),
        json!({ "tag": tag, "pgid": pid, "job": 1 }),
        // A job whose processes are all gone: nothing to stop, and its group number proves nothing.
        json!({ "tag": "gone-tag", "pgid": 1, "start": null, "job": 5 }),
    ]);
    assert_eq!(fates.len(), 1, "{fates:?}");
    assert_eq!(fates[0].job, Some(1));
    assert_eq!(fates[0].survived, None);
    let status = survivor.wait().unwrap();
    assert!(!status.success(), "the survivor was signalled: {status:?}");

    let note = jobs.resume(&transcript()).expect("jobs were left running");
    assert!(
        note.contains("- job 1: `npm run dev` (stopped now)"),
        "{note}"
    );
    assert!(
        note.contains("- job 5: `./serve.sh` (not running)"),
        "{note}"
    );
    assert!(note.contains("killed before it could stop"), "{note}");
}

#[test]
fn a_foreground_command_a_killed_process_left_running_is_named_by_its_command() {
    let jobs = BashJobs::default();
    let tag = format!("reopen-fg-test.{}", std::process::id());
    let mut survivor = tagged_sleep(&tag);
    jobs.stop_survivors(&[json!({
        "tag": tag, "pgid": survivor.id(), "start": null, "command": "cargo test"
    })]);
    let _ = survivor.wait();
    // A transcript that left no job running still gets the note: the command was running.
    let note = jobs
        .resume(&[Message::user("next")])
        .expect("a survivor was stopped");
    assert!(
        note.contains("- `cargo test` (a command still running when it closed; stopped now)"),
        "{note}"
    );
}
