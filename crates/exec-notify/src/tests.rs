#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use serde_json::Value;

const RETRY: &str = include_str!("../../agent/tests/fixtures/exec/v1.4-retry.json");

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

/// Every timeline of `v1.4-retry.json`, driven through [`step`] as an endpoint drives it: the keep
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
        "../../agent/tests/fixtures/exec/v1.4-callback.json"
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

/// Every `v1.4-keep.json` case whose `url` and `token` are strings: the grammar accepts exactly the
/// accepted ones. (A member of the wrong type, or missing, is the caller's to refuse.)
#[test]
fn the_notify_grammar_is_the_vectors() {
    let file: Value = serde_json::from_str(include_str!(
        "../../agent/tests/fixtures/exec/v1.4-keep.json"
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

/// The shell twin ([`SH`]) against [`step`]: every sequence of up to three events, and two thousand
/// longer random ones, through both, comparing every state and every action. Under each shell
/// found here: `sh` (dash on Debian) and busybox `sh` (the `ash` of Alpine and most sandboxes).
#[test]
fn the_shell_kernel_is_step() {
    let events: Vec<Event> = {
        let mut v = vec![
            Event::Keep { notify: true },
            Event::Keep { notify: false },
            Event::Change,
            Event::Removed,
            Event::Wake,
        ];
        for outcome in [
            Outcome::Status(200),
            Outcome::Status(404),
            Outcome::Status(503),
            Outcome::Status(302),
            Outcome::Failed,
        ] {
            v.push(Event::AttemptEnded {
                outcome,
                draw: 0x9e37_79b9,
            });
        }
        v
    };
    let mut seqs: Vec<Vec<(Event, u64)>> = Vec::new();
    let mut prefix = vec![Vec::new()];
    for _ in 0..3 {
        let mut next = Vec::new();
        for p in &prefix {
            for e in &events {
                for gap in [0, 70_000] {
                    let now = p.last().map_or(0, |(_, t): &(Event, u64)| *t) + gap;
                    let mut q: Vec<(Event, u64)> = p.clone();
                    q.push((*e, now));
                    next.push(q);
                }
            }
        }
        seqs.extend(next.iter().cloned());
        prefix = next;
    }
    let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..2000 {
        let mut now = 0;
        let mut q = Vec::new();
        for _ in 0..30 {
            now += [0, 250, 1000, 70_000][(rand() % 4) as usize];
            let mut e = events[(rand() % events.len() as u64) as usize];
            if let Event::AttemptEnded { outcome, .. } = e {
                e = Event::AttemptEnded {
                    outcome,
                    draw: rand() as u32,
                };
            }
            q.push((e, now));
        }
        seqs.push(q);
    }

    // A notifier that died and was started again (its lock removed when it exits, or by the
    // watcher once its pid is gone): a fresh kernel, fed what the supervisor sees on its first
    // pass, the latest keep and then the shell's exit and the end if they are already there, and
    // then its attempts as any notifier's.
    let ended = |outcome, draw| Event::AttemptEnded { outcome, draw };
    seqs.push(vec![
        (Event::Keep { notify: true }, 0),
        (Event::Change, 0),
        (ended(Outcome::Status(503), 7), 120),
        (Event::Wake, 1_120),
        (ended(Outcome::Status(200), 7), 1_300),
    ]);
    seqs.push(vec![
        (Event::Keep { notify: true }, 0),
        (Event::Change, 0),
        (Event::Change, 0),
        (ended(Outcome::Failed, 0xdead_beef), 30_000),
        (Event::Wake, 31_000),
        (ended(Outcome::Status(200), 1), 31_200),
        (Event::Removed, 40_000),
    ]);
    seqs.push(vec![
        (Event::Keep { notify: false }, 0),
        (Event::Change, 500),
    ]);

    let line = |e: &Event, now: u64| match e {
        Event::Keep { notify: true } => format!("k1 {now} - -"),
        Event::Keep { notify: false } => format!("k0 {now} - -"),
        Event::Change => format!("c {now} - -"),
        Event::Removed => format!("r {now} - -"),
        Event::Wake => format!("w {now} - -"),
        Event::AttemptEnded { outcome, draw } => match outcome {
            Outcome::Status(s) => format!("e {now} {s} {draw}"),
            Outcome::Failed => format!("e {now} f {draw}"),
        },
    };
    let state = |j: &Job, acts: &[Action]| {
        let opt = |v: Option<u64>| v.map_or("-".to_string(), |v| v.to_string());
        let mut out = format!(
            "{} {} {} {} {} {} {}|",
            u8::from(j.notify),
            u8::from(j.removed),
            j.owed,
            j.settled,
            opt(j.in_flight),
            j.attempts,
            opt(j.wake_at)
        );
        for a in acts {
            match a {
                Action::Begin { attempt } => out.push_str(&format!("b{attempt} ")),
                Action::Abandon => out.push_str("a "),
                Action::WakeAt(t) => out.push_str(&format!("t{t} ")),
            }
        }
        out
    };
    let mut input = String::new();
    let mut want = String::new();
    for q in &seqs {
        input.push_str("reset 0 - -\n");
        want.push_str("reset\n");
        let mut j = Job::default();
        for (e, now) in q {
            let (n, acts) = step(j, *e, *now);
            j = n;
            input.push_str(&line(e, *now));
            input.push('\n');
            want.push_str(&state(&j, &acts));
            want.push('\n');
        }
    }
    let driver = format!(
        "{SH}\nreset() {{ KN=0 KR=0 KO=0 KS=0 KI= KA=0 KW=; }}\nreset\n\
         while read -r ev now st dr; do\n\
         if [ \"$ev\" = reset ]; then reset; echo reset; continue; fi\n\
         kstep \"$ev\" \"$now\" \"$st\" \"$dr\"\n\
         echo \"$KN $KR $KO $KS ${{KI:--}} $KA ${{KW:--}}|$KX\"\n\
         done\n"
    );
    let mut shells = vec![vec!["sh".to_string()]];
    if std::process::Command::new("busybox")
        .arg("true")
        .status()
        .is_ok_and(|s| s.success())
    {
        shells.push(vec!["busybox".to_string(), "sh".to_string()]);
    }
    for shell in shells {
        use std::io::Write as _;
        let mut child = std::process::Command::new(&shell[0])
            .args(&shell[1..])
            .args(["-c", &driver])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let feed = input.clone();
        let writer = std::thread::spawn(move || stdin.write_all(feed.as_bytes()));
        let out = child.wait_with_output().unwrap();
        writer.join().unwrap().unwrap();
        let got = String::from_utf8(out.stdout).unwrap();
        for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
            assert_eq!(
                g,
                w,
                "{shell:?}: line {i} differs after input {:?}",
                input.lines().nth(i)
            );
        }
        assert_eq!(got.lines().count(), want.lines().count(), "{shell:?}");
    }
}
