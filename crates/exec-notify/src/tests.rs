#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use serde_json::Value;

const RETRY: &str = include_str!("../../sandbox/tests/fixtures/exec/notify-retry.json");

fn outcome(v: &Value) -> Outcome {
    match v.as_u64() {
        Some(s) => Outcome::Status(s as u16),
        None => Outcome::Failed,
    }
}

fn draw(u: f64) -> u32 {
    (u * 4_294_967_296.0).round().min(f64::from(u32::MAX)) as u32
}

#[test]
fn the_status_table_is_the_vectors() {
    let file: Value = serde_json::from_str(RETRY).unwrap();
    for s in file["statuses"].as_array().unwrap() {
        let want = match s["action"].as_str().unwrap() {
            "done" => Verdict::Done,
            "stop" => Verdict::Stop,
            "retry" => Verdict::Retry,
            other => panic!("{other}"),
        };
        assert_eq!(verdict(outcome(&s["answer"])), want, "{}", s["answer"]);
    }
    let sched = &file["schedule"];
    for (i, (max, min)) in sched["waits_max_s"]
        .as_array()
        .unwrap()
        .iter()
        .zip(sched["waits_min_s"].as_array().unwrap())
        .enumerate()
    {
        let n = i as u32 + 1;
        assert_eq!(wait_ms(n, 0), (max.as_f64().unwrap() * 1000.0) as u64);
        assert_eq!(
            wait_ms(n, u32::MAX),
            (min.as_f64().unwrap() * 1000.0) as u64
        );
    }
}

/// Every timeline of `notify-retry.json`, driven through [`step`] as an endpoint drives it: the keep
/// owes the proof, each attempt ends after its duration, and the kernel says when the next begins.
#[test]
fn every_retry_timeline_is_what_step_does() {
    let file: Value = serde_json::from_str(RETRY).unwrap();
    for t in file["timelines"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap();
        let outcomes = t["outcomes"].as_array().unwrap();
        let u = t["u"].as_array().unwrap();
        let (mut j, mut acts) = step(Job::default(), Event::Keep { notify: true }, 0);
        let mut now = 0u64;
        let mut starts = Vec::new();
        let mut ended = 0;
        loop {
            match acts.as_slice() {
                [Action::Begin { attempt }] => {
                    let i = *attempt as usize - 1;
                    starts.push(now);
                    let o = &outcomes[i.min(outcomes.len() - 1)];
                    now += (o["duration_s"].as_f64().unwrap() * 1000.0).round() as u64;
                    ended = now;
                    let d = draw(u.get(i).and_then(Value::as_f64).unwrap_or(0.0));
                    (j, acts) = step(
                        j,
                        Event::AttemptEnded {
                            outcome: outcome(&o["answer"]),
                            draw: d,
                        },
                        now,
                    );
                }
                [Action::WakeAt(at)] => {
                    now = *at;
                    (j, acts) = step(j, Event::Wake, now);
                }
                [] => break,
                other => panic!("{name}: {other:?}"),
            }
        }
        let want = &t["expect"];
        let want_starts: Vec<u64> = want["attempts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| (a["start_s"].as_f64().unwrap() * 1000.0).round() as u64)
            .collect();
        assert_eq!(starts, want_starts, "{name}");
        assert_eq!(
            ended,
            (want["ends_at_s"].as_f64().unwrap() * 1000.0).round() as u64,
            "{name}"
        );
        assert_eq!(
            j.settled, j.owed,
            "{name}: what the count carried is settled"
        );
        assert_eq!(j.in_flight, None);
    }
}

#[test]
fn a_change_during_an_attempt_gets_an_attempt_when_it_ends() {
    let (j, a) = step(Job::default(), Event::Keep { notify: true }, 0);
    assert_eq!(a, [Action::Begin { attempt: 1 }]);
    let (j, a) = step(j, Event::Change, 10);
    assert!(a.is_empty(), "never two attempts in flight");
    let ok = Event::AttemptEnded {
        outcome: Outcome::Status(200),
        draw: 0,
    };
    let (j, a) = step(j, ok, 20);
    assert_eq!(
        a,
        [Action::Begin { attempt: 1 }],
        "a 2xx settles only the proof"
    );
    assert_eq!((j.owed, j.settled), (2, 1));
    let (j, a) = step(j, ok, 30);
    assert!(a.is_empty());
    assert_eq!((j.owed, j.settled, j.in_flight), (2, 2, None));

    // Failed while a change arrived: the next attempt begins at once, a new count.
    let (j, _) = step(Job::default(), Event::Keep { notify: true }, 0);
    let fail = Event::AttemptEnded {
        outcome: Outcome::Status(503),
        draw: 0,
    };
    let (j, a) = step(j, fail, 5);
    assert_eq!(a, [Action::WakeAt(1005)]);
    let (j, a) = step(j, Event::Change, 100);
    assert_eq!(
        a,
        [Action::Begin { attempt: 1 }],
        "a change in a wait begins now"
    );
    let (j, a) = step(j, Event::Change, 110);
    assert!(a.is_empty());
    let (_, a) = step(j, fail, 120);
    assert_eq!(a, [Action::Begin { attempt: 1 }]);
}

#[test]
fn the_latest_keep_governs_and_removal_ends_everything() {
    let (j, _) = step(Job::default(), Event::Keep { notify: true }, 0);
    let (j, a) = step(j, Event::Keep { notify: true }, 1);
    assert_eq!(a, [Action::Abandon, Action::Begin { attempt: 1 }]);
    let (j, a) = step(j, Event::Keep { notify: false }, 2);
    assert_eq!(a, [Action::Abandon]);
    let (j, a) = step(j, Event::Change, 3);
    assert!(a.is_empty(), "no notify, no callback");
    let (j, a) = step(j, Event::Wake, 1_000_000);
    assert!(a.is_empty());
    let (j, _) = step(j, Event::Keep { notify: true }, 4);
    let (j, a) = step(j, Event::Removed, 5);
    assert_eq!(a, [Action::Abandon]);
    for e in [Event::Change, Event::Keep { notify: true }, Event::Wake] {
        assert!(step(j, e, 6).1.is_empty(), "{e:?} after removal");
    }
    // Never kept with notify: a change owes nothing.
    assert!(step(Job::default(), Event::Change, 0).1.is_empty());
}

#[test]
fn a_stop_settles_and_twelve_failures_give_up() {
    let (j, _) = step(Job::default(), Event::Keep { notify: true }, 0);
    let (j, a) = step(
        j,
        Event::AttemptEnded {
            outcome: Outcome::Status(404),
            draw: 0,
        },
        1,
    );
    assert!(a.is_empty());
    assert_eq!(j.settled, j.owed);

    let (mut j, mut a) = step(Job::default(), Event::Keep { notify: true }, 0);
    let mut now = 0;
    let mut attempts = 0;
    loop {
        match a.as_slice() {
            [Action::Begin { attempt }] => {
                attempts = *attempt;
                (j, a) = step(
                    j,
                    Event::AttemptEnded {
                        outcome: Outcome::Failed,
                        draw: 0,
                    },
                    now,
                );
            }
            [Action::WakeAt(at)] => {
                now = *at;
                (j, a) = step(j, Event::Wake, now);
            }
            [] => break,
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(attempts, MAX_ATTEMPTS);
    assert_eq!(now, 363_000, "the twelfth begins at 363 s with u = 0");
    assert_eq!(j.settled, j.owed);
}

/// Every sequence of up to seven events (with two times and two outcomes) from a fresh job: never
/// an attempt beside one in flight, never one without notify or after removal, never an abandon
/// with nothing in flight, and the counts stay consistent.
#[test]
fn every_short_sequence_keeps_the_invariants() {
    let events = [
        Event::Keep { notify: true },
        Event::Keep { notify: false },
        Event::Change,
        Event::Removed,
        Event::AttemptEnded {
            outcome: Outcome::Status(200),
            draw: 0,
        },
        Event::AttemptEnded {
            outcome: Outcome::Failed,
            draw: u32::MAX,
        },
        Event::Wake,
    ];
    fn walk(j: Job, now: u64, depth: u32, events: &[Event]) {
        if depth == 0 {
            return;
        }
        for e in events {
            for later in [0, 70_000] {
                let t = now + later;
                let (n, acts) = step(j, *e, t);
                assert!(n.settled <= n.owed, "{j:?} {e:?} → {n:?}");
                // An attempt's end takes it out of flight by itself.
                let mut flying = j.in_flight.is_some() && !matches!(e, Event::AttemptEnded { .. });
                for a in &acts {
                    match a {
                        Action::Abandon => {
                            assert!(flying, "abandon with nothing in flight: {j:?} {e:?}");
                            flying = false;
                        }
                        Action::Begin { attempt } => {
                            assert!(!flying, "two in flight: {j:?} {e:?}");
                            assert!(n.notify && !n.removed, "{j:?} {e:?} → {n:?}");
                            assert!((1..=MAX_ATTEMPTS).contains(attempt));
                            flying = true;
                        }
                        Action::WakeAt(at) => assert!(*at > t, "{j:?} {e:?}"),
                    }
                }
                assert_eq!(flying, n.in_flight.is_some(), "{j:?} {e:?} → {n:?}");
                if n.in_flight.is_none() && n.wake_at.is_none() && n.notify && !n.removed {
                    assert_eq!(n.owed, n.settled, "owed but nothing scheduled: {n:?}");
                }
                walk(n, t, depth - 1, events);
            }
        }
    }
    walk(Job::default(), 0, 5, &events);
}

#[test]
fn the_body_is_the_vector() {
    let file: Value = serde_json::from_str(include_str!(
        "../../sandbox/tests/fixtures/exec/notify-callback.json"
    ))
    .unwrap();
    assert_eq!(
        body(
            file["keep"]["scope"].as_str().unwrap(),
            file["keep"]["id"].as_str().unwrap()
        ),
        file["requests"][0]["body"].as_str().unwrap()
    );
    assert_eq!(
        body("a\"b\\", "x"),
        r#"{"type":"job_ended","scope":"a\"b\\","id":"x"}"#
    );
}

/// Every `notify-keep.json` case whose `url` and `token` are strings: the grammar accepts exactly the
/// accepted ones. (A member of the wrong type, or missing, is the caller's to refuse.)
#[test]
fn the_notify_grammar_is_the_vectors() {
    let file: Value = serde_json::from_str(include_str!(
        "../../sandbox/tests/fixtures/exec/notify-keep.json"
    ))
    .unwrap();
    let mut checked = 0;
    for case in file["cases"].as_array().unwrap() {
        let n = &case["notify"];
        let (Some(url), Some(token)) = (n["url"].as_str(), n["token"].as_str()) else {
            continue;
        };
        let want = case["expect"]["accepted"].as_bool().unwrap();
        assert_eq!(check_notify(url, token).is_ok(), want, "{}", case["name"]);
        checked += 1;
    }
    assert!(checked > 20);
    for bad in [
        "https://x:99999/",
        "https://ex ample/",
        "https://[::1/",
        "https://x:8a/",
    ] {
        assert!(check_notify(bad, "t").is_err(), "{bad}");
    }
}

/// Production runs the code Verus proves: the three endpoints that decide when to call back all
/// call this crate's `step`, and no other source in the workspace carries a second copy of its
/// state (`Job`'s `wake_at`/`in_flight` pair) or of its answer table.
#[test]
fn production_calls_the_verified_step() {
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let read = |p: &str| std::fs::read_to_string(crates.join(p)).unwrap();
    for caller in ["execd/src/notify.rs", "test-support/src/exec_mock.rs"] {
        assert!(
            read(caller).contains("exec_notify::step_checked("),
            "{caller} must feed events to exec_notify::step_checked (which calls step)"
        );
    }
    assert!(
        read("agent/src/exec_endpoint.rs").contains("pub use exec_notify as notify_kernel;"),
        "the agent's notify_kernel is the shared kernel, not a copy"
    );
    let mut copies = Vec::new();
    let mut dirs = vec![crates.clone()];
    while let Some(d) = dirs.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                if !matches!(
                    p.file_name().and_then(|n| n.to_str()),
                    Some("target" | "node_modules" | ".git")
                ) {
                    dirs.push(p);
                }
            } else if p.extension().is_some_and(|x| x == "rs")
                && !p.starts_with(crates.join("exec-notify"))
                && std::fs::read_to_string(&p).is_ok_and(|s| {
                    s.contains("pub in_flight: Option<u64>") || s.contains("s == 407 || s == 408")
                })
            {
                copies.push(p);
            }
        }
    }
    assert!(
        copies.is_empty(),
        "a second copy of the notify kernel: {copies:?}"
    );
}

/// `step`'s preconditions fail closed at a call site: the clock saturates where the wait cannot
/// wrap, and a job one event from `owed == u64::MAX` is refused one more callback.
#[test]
fn step_checked_never_wraps() {
    assert_eq!(clock(5), 5);
    assert_eq!(clock(u128::MAX), MAX_NOW_MS);
    // Past the last clock step accepts, the clock saturates and the wait does not wrap.
    let (job, _) = step_checked(Job::default(), Event::Keep { notify: true }, 0);
    let ended = Event::AttemptEnded {
        outcome: Outcome::Failed,
        draw: 0,
    };
    let (job, actions) = step_checked(job, ended, u64::MAX);
    assert_eq!(actions, [Action::WakeAt(MAX_NOW_MS + 1_000)]);
    assert!(job.wake_at.is_some());

    // One callback short of the counter's end: the next owes are refused, everything else runs.
    let near = Job {
        notify: true,
        owed: u64::MAX - 1,
        settled: u64::MAX - 1,
        ..Job::default()
    };
    for event in [Event::Keep { notify: true }, Event::Change] {
        let (same, actions) = step_checked(near, event, 0);
        assert_eq!((same, actions.len()), (near, 0), "{event:?}");
    }
    let (off, _) = step_checked(near, Event::Keep { notify: false }, 0);
    assert!(!off.notify);
    assert!(step_checked(near, Event::Removed, 0).0.removed);
    let (below, actions) = step_checked(
        Job {
            owed: u64::MAX - 3,
            settled: u64::MAX - 3,
            ..near
        },
        Event::Change,
        0,
    );
    assert_eq!((below.owed, actions.len()), (u64::MAX - 2, 1));
}
