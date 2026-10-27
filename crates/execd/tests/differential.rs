//! Randomized differential testing: a command run directly on the host vs the same command through
//! a real `execd`, over v1.1 (one JSON answer) and v1.2 (ndjson frames).
//!
//! The example tests and the conformance verifier check the cases somebody thought of. This file
//! generates shell programs that lean on the edges instead (arbitrary bytes on both streams, sizes
//! from 0 to several MiB across the 64 KiB frame and the pipe capacity, stdin, every exit code, a
//! self-kill, background children and process groups) and demands that execd report exactly what a
//! plain `std::process::Command` observes: the bytes of each stream, and the exit code or signal.
//!
//! Deterministic: the generator is a seeded xorshift, so a failure names its seed and reproduces
//! exactly. `EXECD_DIFF_SEED` and `EXECD_DIFF_CASES` move the window locally; the committed run is
//! fixed. A failing program is shrunk (steps dropped, sizes halved) before it is reported.
//!
//! ## What is compared, and what the spec lets differ
//!
//! - **Per stream, not across.** The order of stdout against stderr is best-effort (v1.2
//!   "Ordering"), so each stream is compared on its own.
//! - **Output is bytes.** A one-shot answer's `stdout`/`stderr` are base64, like a stream's frames, so
//!   both are compared with the direct bytes exactly.
//! - **v1.1 signals are `128 + S`** (the convention `shell_exit` documents); the direct run's signal
//!   is mapped the same way. v1.2 carries `{"signal": S}` and is compared exactly.
//! - **stdin is v1.1 only.** A v1.2 command has no stdin, so it is compared against a direct run
//!   with `/dev/null`.
//! - **Background children that outlive the shell.** v1.2 holds the stream open until the group is
//!   empty, so the final output is exact. v1.1 answers when the shell exits, reading pipes for a
//!   100 ms grace, so a late write of a child it did not wait for may be dropped: the answer must
//!   be a prefix of the direct output that is missing at most the background children's bytes.
//! - **Framing** is checked too: no empty output frame, none over 64 KiB decoded, exactly one final
//!   frame and it last.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::io::Write as _;
use std::net::SocketAddr;
use std::os::unix::process::ExitStatusExt as _;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use base64::Engine as _;
use beyond_test_support::ports;
use serde_json::{Value, json};
use support::http::post;

/// xorshift64*, so a seed reproduces a run exactly on any machine.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// The size of the shared file of random bytes that `head -c` reads from.
const POOL: usize = 5 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Stream {
    Out,
    Err,
}

impl Stream {
    fn redirect(self) -> &'static str {
        match self {
            Stream::Out => "",
            Stream::Err => " >&2",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Stdin {
    Cat,
    Wc,
    Od,
}

#[derive(Clone, Debug)]
enum Step {
    /// `printf` of every byte as a 3-digit octal escape.
    Printf(Stream, Vec<u8>),
    /// `tail -c +skip pool | head -c len`: `len` seeded random bytes.
    Pool { to: Stream, skip: usize, len: usize },
    /// A stdin consumer.
    Stdin(Stdin, Stream),
    /// A background child, optionally in a process group of its own (`set -m`), that writes after
    /// `ms` milliseconds; `wait` joins it before the script goes on.
    Background {
        to: Stream,
        ms: u32,
        wait: bool,
        group: bool,
        nested: bool,
    },
    /// `$HOME` and `pwd`, the conventions both sides must share.
    Env,
}

#[derive(Clone, Debug)]
enum End {
    Exit(u8),
    Kill(&'static str),
}

#[derive(Clone, Debug)]
struct Program {
    steps: Vec<Step>,
    end: End,
    stdin: Option<Vec<u8>>,
}

const SIGNALS: [&str; 4] = ["TERM", "KILL", "INT", "HUP"];

/// Byte values that decide behavior: NUL, newline and CR, ESC, a lone UTF-8 continuation, a
/// truncated sequence, DEL, 0xFF, and the printf metacharacters.
const NASTY: [u8; 12] = [
    0x00, 0x0a, 0x0d, 0x1b, 0x80, 0xc3, 0xe2, 0x7f, 0xff, b'%', b'\\', b'\'',
];

fn size(rng: &mut Rng, big: &mut bool) -> usize {
    const EDGES: [usize; 12] = [
        0, 1, 2, 4095, 4096, 65535, 65536, 65537, 131_071, 131_072, 131_073, 300_000,
    ];
    match rng.below(10) {
        0..=3 => EDGES[rng.below(EDGES.len())],
        4..=6 => rng.below(70_000),
        7 | 8 => rng.below(400_000),
        // Several MiB, well past the 64 KiB pipe, at most once per program.
        _ if !*big => {
            *big = true;
            1_000_000 + rng.below(3_500_000)
        }
        _ => rng.below(1000),
    }
}

fn stream(rng: &mut Rng) -> Stream {
    if rng.chance(2) {
        Stream::Out
    } else {
        Stream::Err
    }
}

fn generate(seed: u64) -> Program {
    let mut rng = Rng::new(seed);
    let mut big = false;
    let mut stdin_step = false;
    let mut steps = Vec::new();
    for _ in 0..1 + rng.below(7) {
        let step = match rng.below(10) {
            0..=2 => {
                let len = rng.below(80);
                let mut b = rng.bytes(len);
                for _ in 0..rng.below(6) {
                    if !b.is_empty() {
                        let i = rng.below(b.len());
                        b[i] = NASTY[rng.below(NASTY.len())];
                    }
                }
                Step::Printf(stream(&mut rng), b)
            }
            3..=5 => Step::Pool {
                to: stream(&mut rng),
                skip: 1 + rng.below(POOL / 8),
                len: size(&mut rng, &mut big),
            },
            // One consumer per program: two `head`s would split the stdin pipe by timing.
            6 | 7 if !stdin_step => {
                stdin_step = true;
                let c = [Stdin::Cat, Stdin::Wc, Stdin::Od][rng.below(3)];
                Step::Stdin(c, stream(&mut rng))
            }
            8 => Step::Background {
                to: stream(&mut rng),
                ms: 5 + rng.below(30) as u32,
                wait: rng.chance(2),
                group: rng.chance(2),
                nested: rng.chance(3),
            },
            _ => Step::Env,
        };
        steps.push(step);
    }
    let end = match rng.below(6) {
        0 => End::Kill(SIGNALS[rng.below(SIGNALS.len())]),
        1 => End::Exit(0),
        2 => End::Exit([1, 2, 126, 127, 128, 129, 130, 137, 143, 254, 255][rng.below(11)]),
        _ => End::Exit(rng.below(256) as u8),
    };
    // stdin goes to the command at random whether or not it reads it.
    let stdin = (stdin_step || rng.chance(4)).then(|| {
        let n = match rng.below(5) {
            0 => 0,
            1 => rng.below(100),
            2 => 65536 + rng.below(3),
            _ => rng.below(1_200_000),
        };
        rng.bytes(n)
    });
    Program { steps, end, stdin }
}

impl Program {
    fn script(&self) -> String {
        // A child the shell does not wait for is launched last, after every byte of the shell's own
        // output is in the pipe: two writers racing on one stream have no defined order, and that
        // is not what is under test.
        let mut s = String::new();
        let mut late = String::new();
        if self
            .steps
            .iter()
            .any(|s| matches!(s, Step::Background { group: true, .. }))
        {
            s += "set -m\n";
        }
        for step in &self.steps {
            match step {
                Step::Printf(to, b) => {
                    let esc: String = b.iter().map(|x| format!("\\{x:03o}")).collect();
                    s += &format!("printf '{esc}'{}\n", to.redirect());
                }
                Step::Pool { to, skip, len } => {
                    s += &format!("tail -c +{skip} pool | head -c {len}{}\n", to.redirect());
                }
                Step::Stdin(c, to) => {
                    let cmd = match c {
                        Stdin::Cat => "cat",
                        Stdin::Wc => "wc -c",
                        Stdin::Od => "od -An -tx1",
                    };
                    s += &format!("{cmd}{}\n", to.redirect());
                }
                Step::Background {
                    to,
                    ms,
                    wait,
                    nested,
                    ..
                } => {
                    let secs = format!("0.{ms:03}");
                    let body = if *nested {
                        format!("sh -c 'sleep {secs}; printf y{}' &", to.redirect())
                    } else {
                        format!("sleep {secs}; printf x{}", to.redirect())
                    };
                    // `wait` does not join a grandchild: the subshell exits and leaves it running.
                    if *wait && !*nested {
                        s += &format!("({body}) &\nwait\n");
                    } else {
                        late += &format!("({body}) &\n");
                    }
                }
                Step::Env => s += "printf '%s|' \"$HOME\"; pwd\n",
            }
        }
        s += &late;
        match &self.end {
            End::Exit(n) => s += &format!("exit {n}\n"),
            End::Kill(sig) => s += &format!("kill -{sig} $$\n"),
        }
        s
    }

    /// How many background children the shell does not wait for: each may add one byte after the
    /// shell exits.
    fn loose_bytes(&self, on: Stream) -> usize {
        self.steps
            .iter()
            .filter(|s| {
                matches!(s, Step::Background { to, wait, nested, .. }
                    if *to == on && (!*wait || *nested))
            })
            .count()
    }
}

#[derive(Debug, PartialEq)]
enum Ended {
    Exit(i32),
    Signal(i32),
}

struct Outcome {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    ended: Ended,
}

/// The command on the host, as execd runs it: same cwd, same `HOME`, piped output.
fn direct(root: &Path, home: &Path, script: &str, stdin: Option<&[u8]>) -> Outcome {
    let mut c = Command::new("sh");
    c.args(["-c", script])
        .current_dir(root)
        .env("HOME", home)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = c.spawn().unwrap();
    let feeder = stdin.map(|b| {
        let mut pipe = child.stdin.take().unwrap();
        let b = b.to_vec();
        std::thread::spawn(move || {
            let _ = pipe.write_all(&b);
        })
    });
    let out = child.wait_with_output().unwrap();
    if let Some(f) = feeder {
        f.join().unwrap();
    }
    Outcome {
        stdout: out.stdout,
        stderr: out.stderr,
        ended: match (out.status.code(), out.status.signal()) {
            (Some(c), _) => Ended::Exit(c),
            (None, Some(s)) => Ended::Signal(s),
            _ => unreachable!(),
        },
    }
}

async fn direct_async(root: &Path, home: &Path, script: &str, stdin: Option<Vec<u8>>) -> Outcome {
    let (r, h, s) = (root.to_owned(), home.to_owned(), script.to_owned());
    tokio::task::spawn_blocking(move || direct(&r, &h, &s, stdin.as_deref()))
        .await
        .unwrap()
}

fn request(script: &str) -> Value {
    json!({ "command": "sh", "args": ["-c", script], "timeout_ms": 60_000 })
}

fn decode(v: &Value) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(v.as_str().expect("base64 output"))
        .expect("standard base64")
}

/// One-shot: one JSON answer. `128 + S` stands for a signal here, so it is reported as an exit code.
async fn via_v11(addr: SocketAddr, script: &str, stdin: Option<&[u8]>) -> Outcome {
    let mut req = request(script);
    if let Some(b) = stdin {
        req["stdin"] = json!(base64::engine::general_purpose::STANDARD.encode(b));
    }
    let a = post(addr, &[], &req).await;
    assert_eq!(a.status, 200, "v1.1 status");
    let j = a.json();
    Outcome {
        stdout: decode(&j["stdout"]),
        stderr: decode(&j["stderr"]),
        ended: Ended::Exit(j["exit_code"].as_i64().expect("exit_code") as i32),
    }
}

/// v1.2: frames, decoded and concatenated per stream, with the framing rules checked.
async fn via_v12(addr: SocketAddr, script: &str) -> Result<Outcome, String> {
    let mut req = request(script);
    req["stream"] = json!(true);
    let a = post(addr, &[], &req).await;
    if a.status != 200 {
        return Err(format!("v1.2 status {}", a.status));
    }
    let mut o = Outcome {
        stdout: Vec::new(),
        stderr: Vec::new(),
        ended: Ended::Exit(-1),
    };
    let mut finals = 0;
    for (i, f) in a.frames().iter().enumerate() {
        if finals > 0 {
            return Err(format!("frame after the final one: {f}"));
        }
        for (key, into) in [("stdout", &mut o.stdout), ("stderr", &mut o.stderr)] {
            if let Some(b64) = f.get(key).and_then(Value::as_str) {
                let b = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| format!("frame {i}: bad base64: {e}"))?;
                if b.is_empty() || b.len() > 64 * 1024 {
                    return Err(format!("frame {i}: {key} frame of {} bytes", b.len()));
                }
                into.extend_from_slice(&b);
            }
        }
        if let Some(c) = f.get("exit_code").and_then(Value::as_i64) {
            o.ended = Ended::Exit(c as i32);
            finals += 1;
        } else if let Some(s) = f.get("signal").and_then(Value::as_i64) {
            o.ended = Ended::Signal(s as i32);
            finals += 1;
        } else if f.get("timed_out").is_some() || f.get("error").is_some() {
            return Err(format!("unexpected final frame {f}"));
        }
    }
    if finals != 1 {
        return Err(format!("{finals} final frames"));
    }
    Ok(o)
}

fn first_diff(a: &[u8], b: &[u8]) -> String {
    let at = a
        .iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()));
    format!(
        "len {} vs {}, first difference at byte {at}",
        a.len(),
        b.len()
    )
}

/// Run `p` every way and compare. `Err` names the first disagreement.
async fn check(addr: SocketAddr, root: &Path, home: &Path, p: &Program) -> Result<(), String> {
    let script = p.script();

    // v1.2: no stdin on either side.
    let want = direct_async(root, home, &script, None).await;
    let got = via_v12(addr, &script).await?;
    for (name, on, g, w) in [
        ("stdout", Stream::Out, &got.stdout, &want.stdout),
        ("stderr", Stream::Err, &got.stderr, &want.stderr),
    ] {
        same(g, w, p.loose_bytes(on), true).map_err(|e| format!("v1.2 {name}: {e}"))?;
    }
    if got.ended != want.ended {
        return Err(format!(
            "v1.2 end: {:?} vs direct {:?}",
            got.ended, want.ended
        ));
    }

    // v1.1, with the stdin the program carries.
    let want = direct_async(root, home, &script, p.stdin.clone()).await;
    let got = via_v11(addr, &script, p.stdin.as_deref()).await;
    let want_code = match want.ended {
        Ended::Exit(c) => c,
        Ended::Signal(s) => 128 + s,
    };
    if got.ended != Ended::Exit(want_code) {
        return Err(format!(
            "v1.1 exit_code: {:?} vs direct {want_code}",
            got.ended
        ));
    }
    for (name, on, g, w) in [
        ("stdout", Stream::Out, &got.stdout, &want.stdout),
        ("stderr", Stream::Err, &got.stderr, &want.stderr),
    ] {
        same(g, w, p.loose_bytes(on), false).map_err(|e| format!("v1.1 {name}: {e}"))?;
    }
    Ok(())
}

/// `got` is `want`, except that its last `loose` bytes belong to background children the shell did
/// not wait for. Those arrive in whatever order the scheduler gives them, so only their multiset
/// is compared; and when the endpoint may answer before they write (`exact` is false: v1.1) some
/// may be missing.
fn same(got: &[u8], want: &[u8], loose: usize, exact: bool) -> Result<(), String> {
    let head = want.len().saturating_sub(loose);
    let long_enough = if exact {
        got.len() == want.len()
    } else {
        got.len() >= head && got.len() <= want.len()
    };
    if !long_enough || got[..head] != want[..head] {
        return Err(first_diff(got, want));
    }
    let mut left = [0usize; 256];
    for &b in &want[head..] {
        left[b as usize] += 1;
    }
    for &b in &got[head..] {
        match left[b as usize].checked_sub(1) {
            Some(n) => left[b as usize] = n,
            None => {
                return Err(format!(
                    "unexpected late byte {b:#x}; {}",
                    first_diff(got, want)
                ));
            }
        }
    }
    Ok(())
}

/// Drop steps and halve sizes while the program still fails.
async fn shrink(addr: SocketAddr, root: &Path, home: &Path, mut p: Program) -> Program {
    loop {
        let mut cands: Vec<Program> = Vec::new();
        for i in 0..p.steps.len() {
            let mut q = p.clone();
            q.steps.remove(i);
            cands.push(q);
        }
        for i in 0..p.steps.len() {
            let mut q = p.clone();
            match &mut q.steps[i] {
                Step::Pool { len, skip, .. } if *len > 0 || *skip > 1 => {
                    *len /= 2;
                    *skip = 1;
                }
                Step::Printf(_, b) if b.len() > 1 => b.truncate(b.len() / 2),
                _ => continue,
            }
            cands.push(q);
        }
        if let Some(s) = p.stdin.as_ref().filter(|s| !s.is_empty()) {
            let mut q = p.clone();
            q.stdin = Some(s[..s.len() / 2].to_vec());
            cands.push(q);
        }
        let mut next = None;
        for q in cands {
            if check(addr, root, home, &q).await.is_err() {
                next = Some(q);
                break;
            }
        }
        match next {
            Some(q) => p = q,
            None => return p,
        }
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn execd_agrees_with_the_host_on_random_programs() {
    let root = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    let home_path = home.path().canonicalize().unwrap();
    std::fs::write(root_path.join("pool"), Rng::new(0xC0FFEE).bytes(POOL)).unwrap();
    let mut o = execd::Options::new(&root_path);
    o.home = Some(home_path.clone());
    let server = execd::serve(ports::tokio_listener().await, o).unwrap();
    let addr = server.local_addr();

    let first = env_u64("EXECD_DIFF_SEED", 1);
    let cases = env_u64("EXECD_DIFF_CASES", 40);
    let started = Instant::now();
    for seed in first..first + cases {
        let p = generate(seed);
        if let Err(why) = check(addr, &root_path, &home_path, &p).await {
            let small = shrink(addr, &root_path, &home_path, p).await;
            let again = check(addr, &root_path, &home_path, &small).await.err();
            panic!(
                "execd disagrees with the host. Reproduce: EXECD_DIFF_SEED={seed} EXECD_DIFF_CASES=1\n\
                 first failure: {why}\nshrunk program: {:?}\nstdin: {} bytes\nscript:\n{}\nshrunk failure: {again:?}",
                small.steps,
                small.stdin.as_ref().map_or(0, Vec::len),
                small.script(),
            );
        }
    }
    eprintln!(
        "{cases} programs from seed {first} agreed in {:?}",
        started.elapsed()
    );
    server.shutdown().await;
}
