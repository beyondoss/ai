//! The v1.3 sandbox-job decoding vectors (`tests/fixtures/exec/v1.3-jobs.json`): job stream frames
//! and the answers to non-streamed job operations, read the way the agent reads them. An endpoint
//! implementer can check their encoder against the same file.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use base64::Engine as _;
use beyond_agent::exec_endpoint::jobs::JobStatus;
use beyond_agent::exec_endpoint::jobs_native::{
    Event, JobFrame, decode_job_frame, parse_list, parse_read, parse_survivors,
};
use serde_json::Value;

fn vectors() -> Value {
    serde_json::from_str(include_str!("fixtures/exec/v1.3-jobs.json")).unwrap()
}

fn b64(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s).unwrap()
}

fn state_name(s: JobStatus) -> (&'static str, Option<i64>) {
    match s {
        JobStatus::Running => ("running", None),
        JobStatus::ShellExited(c) => ("exited", Some(c.into())),
        JobStatus::Ended(c) => ("ended", Some(c.into())),
        JobStatus::Lost => ("lost", None),
        JobStatus::Gone => ("gone", None),
    }
}

#[test]
fn job_streams_decode_as_the_vectors_say() {
    for case in vectors()["streams"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (mut started, mut output, mut gap, mut shell) = (None, Vec::new(), 0, Vec::new());
        let (mut end, mut error) = (None, None);
        for line in case["lines"].as_array().unwrap() {
            match decode_job_frame(line.as_str().unwrap().as_bytes()).unwrap() {
                JobFrame::Event(Event::Started(p)) => started = Some(p),
                JobFrame::Event(Event::Output(b)) => output.extend(b),
                JobFrame::Event(Event::Gap(n)) => gap += n,
                JobFrame::Event(Event::ShellExit(c)) => shell.push(c),
                JobFrame::Heartbeat => {}
                JobFrame::End(c) => end = Some(c),
                JobFrame::Error(e) => error = Some(e),
            }
        }
        let want = &case["expect"];
        assert_eq!(started.map(u64::from), want["started"].as_u64(), "{name}");
        assert_eq!(
            output,
            b64(want["output_base64"].as_str().unwrap()),
            "{name}"
        );
        assert_eq!(gap, want["gap"].as_u64().unwrap(), "{name}");
        let shells: Vec<i32> = want["shell_exits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(shell, shells, "{name}");
        assert_eq!(end.map(i64::from), want["end"].as_i64(), "{name}");
        assert_eq!(error.as_deref(), want["error"].as_str(), "{name}");
    }
}

#[test]
fn job_answers_read_as_the_vectors_say() {
    for case in vectors()["answers"].as_array().unwrap() {
        let (answer, want) = (&case["answer"], &case["expect"]);
        match case["op"].as_str().unwrap() {
            "read" => {
                let r = parse_read(answer).unwrap();
                let (state, code) = state_name(r.status);
                assert_eq!(state, want["state"].as_str().unwrap());
                assert_eq!(code, want["code"].as_i64());
                assert_eq!(r.offset, want["offset"].as_u64().unwrap());
                assert_eq!(r.end, want["end"].as_u64().unwrap());
                assert_eq!(r.data, b64(want["data_base64"].as_str().unwrap()));
            }
            "list" => {
                let jobs = parse_list(answer).unwrap();
                let wanted = want["jobs"].as_array().unwrap();
                assert_eq!(jobs.len(), wanted.len(), "an id that is not one is skipped");
                for (j, w) in jobs.iter().zip(wanted) {
                    let (state, code) = state_name(j.status);
                    assert_eq!(j.id.as_str(), w["id"].as_str().unwrap());
                    assert_eq!(state, w["state"].as_str().unwrap());
                    assert_eq!(code, w["code"].as_i64());
                    assert_eq!(j.kept, w["kept"].as_bool().unwrap());
                    assert_eq!(j.num, w["num"].as_u64());
                    assert_eq!(j.kind.as_deref(), w["kind"].as_str());
                    assert_eq!(j.seen, w["seen"].as_u64().unwrap());
                    assert_eq!(j.ended, w["ended"].as_u64().unwrap());
                    assert_eq!(j.label, w["label"].as_str().unwrap());
                }
            }
            "kill" => {
                let s = parse_survivors(answer);
                let w = &want["survivors"][0];
                assert_eq!(s.len(), 1);
                assert_eq!(s[0].0.as_deref(), w["id"].as_str());
                assert_eq!(s[0].1, w["described"].as_str().unwrap());
            }
            other => panic!("no vector reader for {other}"),
        }
    }
}
