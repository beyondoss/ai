#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use std::sync::atomic::AtomicU64;

fn dir(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "usage-file-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// 2026-10-17T10:00:00Z.
const T0: u64 = 1_792_231_200;

fn clock(secs: &Arc<AtomicU64>) -> Clock {
    let secs = Arc::clone(secs);
    Box::new(move || UNIX_EPOCH + Duration::from_secs(secs.load(Ordering::Relaxed)))
}

fn open(d: &Path, secs: &Arc<AtomicU64>) -> (UsageFiles, IntCounter) {
    let errors = IntCounter::new("e", "e").unwrap();
    let f = UsageFiles::open_with(d, errors.clone(), clock(secs), None).unwrap();
    (f, errors)
}

fn names(d: &Path) -> Vec<String> {
    let mut v: Vec<_> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    v.sort();
    v
}

fn read(d: &Path, name: &str) -> String {
    std::fs::read_to_string(d.join(name)).unwrap()
}

#[test]
fn names_are_utc_hours_and_round_trip() {
    assert_eq!(stem(T0 / 3600), "ai-usage-20261017T10");
    assert_eq!(stem(0), "ai-usage-19700101T00");
    assert_eq!(parse_hour("ai-usage-20261017T10.jsonl"), Some(T0 / 3600));
    assert_eq!(
        parse_hour("ai-usage-20261017T10.4242.jsonl"),
        Some(T0 / 3600)
    );
    for bad in [
        "ai-usage-20261017T10.txt",
        "other.jsonl",
        "ai-usage-20261317T10.jsonl",
        "ai-usage-20261017T25.jsonl",
        "ai-usage-20261017T10.x.jsonl",
    ] {
        assert_eq!(parse_hour(bad), None, "{bad}");
    }
    for h in [0, 1, 23, 24, 17_000 * 24 + 5, T0 / 3600] {
        assert_eq!(parse_hour(&format!("{}.jsonl", stem(h))), Some(h));
    }
}

/// Crossing the hour mid-stream: every row lands once, in the file of the hour it was written in.
#[test]
fn rotation_loses_and_duplicates_nothing() {
    let d = dir("rot");
    let secs = Arc::new(AtomicU64::new(T0 + 3590));
    let (f, _) = open(&d, &secs);
    for i in 0..10u64 {
        f.write_row(format!("{{\"n\":{i}}}\n").as_bytes());
        secs.fetch_add(2, Ordering::Relaxed); // crosses 11:00 after row 4
    }
    assert_eq!(
        names(&d),
        ["ai-usage-20261017T10.jsonl", "ai-usage-20261017T11.jsonl"]
    );
    let a = read(&d, "ai-usage-20261017T10.jsonl");
    let b = read(&d, "ai-usage-20261017T11.jsonl");
    let all: Vec<String> = a.lines().chain(b.lines()).map(str::to_owned).collect();
    let want: Vec<String> = (0..10).map(|i| format!("{{\"n\":{i}}}")).collect();
    assert_eq!(all, want);
    assert_eq!(a.lines().count(), 5);
    assert_eq!(b.lines().count(), 5);
}

/// Rows written through the `MakeWriter` in several `write` calls stay whole lines.
#[test]
fn a_row_written_in_pieces_lands_as_one_line() {
    let d = dir("pieces");
    let secs = Arc::new(AtomicU64::new(T0));
    let (f, _) = open(&d, &secs);
    {
        let mut w = f.make_writer();
        w.write_all(b"{\"a\":").unwrap();
        w.write_all(b"1}\n").unwrap();
    }
    assert_eq!(read(&d, "ai-usage-20261017T10.jsonl"), "{\"a\":1}\n");
}

#[test]
fn retention_deletes_past_48h_and_keeps_recent_and_foreign_files() {
    let d = dir("ret");
    std::fs::create_dir_all(&d).unwrap();
    let hour = T0 / 3600;
    let old = stem(hour - 49);
    let edge = stem(hour - 48);
    let recent = stem(hour - 47);
    for s in [&old, &edge, &recent] {
        std::fs::write(d.join(format!("{s}.jsonl")), "x\n").unwrap();
    }
    std::fs::write(d.join(format!("{old}.77.jsonl")), "x\n").unwrap();
    std::fs::write(d.join("notes.txt"), "x").unwrap();
    std::fs::write(d.join("other.jsonl"), "x").unwrap();
    let secs = Arc::new(AtomicU64::new(T0));
    let (f, _) = open(&d, &secs);
    f.write_row(b"{}\n");
    let got = names(&d);
    assert!(!got.contains(&format!("{old}.jsonl")), "{got:?}");
    assert!(!got.contains(&format!("{old}.77.jsonl")), "{got:?}");
    for keep in [&edge, &recent, &stem(hour)] {
        assert!(got.contains(&format!("{keep}.jsonl")), "{keep} in {got:?}");
    }
    assert!(got.contains(&"notes.txt".to_string()) && got.contains(&"other.jsonl".to_string()));
}

/// The current and previous hours survive a sweep however far the clock has moved.
#[test]
fn retention_never_deletes_current_or_previous_hour() {
    let d = dir("ret2");
    let secs = Arc::new(AtomicU64::new(T0));
    let (f, _) = open(&d, &secs);
    f.write_row(b"1\n");
    secs.fetch_add(3600, Ordering::Relaxed);
    f.write_row(b"2\n");
    secs.fetch_add(3600, Ordering::Relaxed);
    f.write_row(b"3\n");
    assert_eq!(names(&d).len(), 3);
    sweep(&d, T0 / 3600 + 2);
    assert_eq!(names(&d).len(), 3);
    // Far in the future: only the previous hour (T0+1h) and the current one (T0+2h) are safe.
    sweep(&d, T0 / 3600 + 2 + 1000);
    assert_eq!(names(&d).len(), 0);
    f.write_row(b"4\n");
}

#[test]
fn a_restart_appends_and_ends_a_torn_line() {
    let d = dir("restart");
    let secs = Arc::new(AtomicU64::new(T0));
    {
        let (f, _) = open(&d, &secs);
        f.write_row(b"{\"n\":1}\n");
    }
    // A predecessor killed mid-row.
    let p = d.join("ai-usage-20261017T10.jsonl");
    let mut fh = OpenOptions::new().append(true).open(&p).unwrap();
    fh.write_all(b"{\"n\":2").unwrap();
    drop(fh);
    let (f, _) = open(&d, &secs);
    f.write_row(b"{\"n\":3}\n");
    assert_eq!(
        read(&d, "ai-usage-20261017T10.jsonl"),
        "{\"n\":1}\n{\"n\":2\n{\"n\":3}\n"
    );
}

/// A second writer in the same hour does not share the first's file.
#[test]
fn a_second_writer_gets_its_own_file() {
    let d = dir("two");
    let secs = Arc::new(AtomicU64::new(T0));
    let (a, _) = open(&d, &secs);
    let (b, _) = open(&d, &secs);
    a.write_row(b"a\n");
    b.write_row(b"b\n");
    let n = names(&d);
    assert_eq!(n.len(), 2, "{n:?}");
    assert_eq!(read(&d, "ai-usage-20261017T10.jsonl"), "a\n");
    assert!(
        n.iter()
            .any(|x| x.starts_with("ai-usage-20261017T10.") && x != "ai-usage-20261017T10.jsonl")
    );
}

#[test]
fn an_unwritable_directory_fails_at_open() {
    let d = dir("bad");
    std::fs::create_dir_all(&d).unwrap();
    let blocker = d.join("file");
    std::fs::write(&blocker, "x").unwrap();
    let secs = Arc::new(AtomicU64::new(T0));
    let r = UsageFiles::open_with(
        &blocker.join("sub"),
        IntCounter::new("e", "e").unwrap(),
        clock(&secs),
        None,
    );
    assert!(r.is_err());
}

/// A write that cannot land blocks the writer and is counted; the row lands, once, when the fault
/// clears. The fault is a directory squatting on the next hour's file name (fails even as root).
#[test]
fn a_failing_write_blocks_then_lands_the_row_exactly_once() {
    let d = dir("bp");
    let secs = Arc::new(AtomicU64::new(T0));
    let (f, errors) = open(&d, &secs);
    f.write_row(b"before\n");
    secs.fetch_add(3600, Ordering::Relaxed);
    let squat = d.join("ai-usage-20261017T11.jsonl");
    std::fs::create_dir(&squat).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let f2 = f.clone();
    let t = std::thread::spawn(move || {
        f2.write_row(b"after\n");
        tx.send(()).unwrap();
    });
    assert!(
        rx.recv_timeout(Duration::from_millis(500)).is_err(),
        "must block, not drop"
    );
    assert_eq!(errors.get(), 1);
    std::fs::remove_dir(&squat).unwrap();
    rx.recv_timeout(Duration::from_secs(5))
        .expect("lands once the fault clears");
    t.join().unwrap();
    assert_eq!(read(&d, "ai-usage-20261017T10.jsonl"), "before\n");
    assert_eq!(read(&d, "ai-usage-20261017T11.jsonl"), "after\n");
    assert_eq!(errors.get(), 1);
}

/// Measurement: per-row cost of the write path on this host (run with `--nocapture`).
#[test]
fn per_row_cost_is_a_write_syscall() {
    let d = dir("cost");
    let secs = Arc::new(AtomicU64::new(T0));
    let (f, _) = open(&d, &secs);
    let mut row = vec![b'x'; 399];
    row.push(b'\n');
    let n = 20_000;
    let t = std::time::Instant::now();
    for _ in 0..n {
        f.write_row(&row);
    }
    let per_row = t.elapsed() / n;
    eprintln!("usage file write_row: {per_row:?} per 400-byte row");
    f.sync();
    let t = std::time::Instant::now();
    for _ in 0..200 {
        f.write_row(&row);
        f.sync();
    }
    eprintln!(
        "usage file write_row+fdatasync: {:?} per row",
        t.elapsed() / 200
    );
    assert!(per_row < Duration::from_millis(1));
}
