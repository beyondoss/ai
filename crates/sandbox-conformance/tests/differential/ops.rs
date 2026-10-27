//! The operations a generated sequence is made of, and the Hegel generator that draws them.
//!
//! An [`Op`] is one DRIVER.md request with paths relative to the case's root. [`Op::step`] renders it
//! as a scenario step whose strings may contain `${ROOT}`; the runner substitutes each driver's own
//! root, and a failing sequence is printed as the same steps, so what ran is what is reported.
//!
//! The name, content and query generators are the ones of the `ShellFs` property test
//! (`crates/agent/tests/shellfs_props_env/fixtures.rs`), included, not copied.
use base64::Engine as _;
use hegel::TestCase;
use hegel::generators as gs;
use serde_json::{Value, json};

#[path = "../../../agent/tests/shellfs_props_env/fixtures.rs"]
mod fixtures;

#[derive(Clone, Debug)]
pub enum Guard {
    /// `expected_mtime: null`: the file must not exist.
    Absent,
    /// An mtime no file has.
    Stale,
    /// The mtime the `Stat` at this op index reported (null when it reported no file).
    Current(usize),
}

#[derive(Clone, Debug)]
pub enum Op {
    Write {
        rel: String,
        data: Vec<u8>,
    },
    WriteIf {
        rel: String,
        data: Vec<u8>,
        guard: Guard,
    },
    Read {
        rel: String,
        offset: u64,
        max: Option<u64>,
    },
    Stat {
        rel: String,
    },
    List {
        rel: String,
        hidden: bool,
        limit: Option<usize>,
    },
    Search {
        rel: String,
        pattern: String,
        literal: bool,
        ignore_case: bool,
        glob: Option<(String, bool)>,
        limit: usize,
        before: usize,
        after: usize,
        /// A pattern outside the subset the oracle's matcher models (see `search_truth`): checked
        /// against the other drivers only.
        raw: bool,
    },
    Mkdir {
        rel: String,
    },
    /// A raw exec: the environment steps (a change behind the client's back) are these too.
    Exec {
        command: String,
        args: Vec<String>,
        cwd: Option<String>,
        stdin: Option<Vec<u8>>,
        timeout_ms: Option<u64>,
        stream: bool,
    },
}

fn b64(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

/// The base64 of `data` for a scenario file: a long periodic payload becomes a `$repeat`, so a
/// printed repro of a 150 KB write is a few lines.
fn data_value(data: &[u8]) -> Value {
    if data.len() > 1024
        && let Some(p) = (1..=25).find(|p| data.iter().enumerate().all(|(i, b)| *b == data[i % p]))
    {
        return json!({"$repeat": {"b64": b64(&data[..p]), "times": data.len().div_ceil(p), "len": data.len()}});
    }
    json!(b64(data))
}

pub fn abs(rel: &str) -> String {
    if rel.is_empty() {
        "${ROOT}".to_string()
    } else {
        format!("${{ROOT}}/{rel}")
    }
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::Write { .. } => "write",
            Op::WriteIf { .. } => "write_if_unchanged",
            Op::Read { .. } => "read",
            Op::Stat { .. } => "stat",
            Op::List { .. } => "list",
            Op::Search { .. } => "search",
            Op::Mkdir { .. } => "mkdir",
            Op::Exec { stream: false, .. } => "exec",
            Op::Exec { stream: true, .. } => "exec_stream",
        }
    }

    /// The request, minus `id`. `expected_mtime` of a `Current` guard is `"${m<i>}"`; the runner
    /// swaps in the value its own driver's `Stat` returned.
    pub fn step(&self) -> Value {
        let mut v = match self {
            Op::Write { rel, data } => json!({"path": abs(rel), "data_b64": data_value(data)}),
            Op::WriteIf { rel, data, guard } => json!({
                "path": abs(rel),
                "data_b64": data_value(data),
                "expected_mtime": match guard {
                    Guard::Absent => Value::Null,
                    Guard::Stale => json!(1),
                    Guard::Current(i) => json!(format!("${{m{i}}}")),
                },
            }),
            Op::Read { rel, offset, max } => {
                let mut v = json!({"path": abs(rel), "offset": offset});
                if let Some(m) = max {
                    v["max"] = json!(m);
                }
                v
            }
            Op::Stat { rel } | Op::Mkdir { rel } => json!({"path": abs(rel)}),
            Op::List { rel, hidden, limit } => {
                let mut v = json!({"path": abs(rel), "include_hidden": hidden});
                if let Some(l) = limit {
                    v["limit"] = json!(l);
                }
                v
            }
            Op::Search {
                rel,
                pattern,
                literal,
                ignore_case,
                glob,
                limit,
                before,
                after,
                ..
            } => {
                let mut v = json!({"pattern": pattern, "root": abs(rel), "literal": literal,
                    "ignore_case": ignore_case, "limit": limit, "before": before, "after": after});
                if let Some((g, ex)) = glob {
                    v["glob"] = json!(g);
                    v["glob_exclude"] = json!(ex);
                }
                v
            }
            Op::Exec {
                command,
                args,
                cwd,
                stdin,
                timeout_ms,
                ..
            } => {
                let mut v = json!({"command": command, "args": args});
                if let Some(c) = cwd {
                    v["cwd"] = json!(c);
                }
                if let Some(s) = stdin {
                    v["stdin_b64"] = data_value(s);
                }
                if let Some(t) = timeout_ms {
                    v["timeout_ms"] = json!(t);
                }
                v
            }
        };
        v["op"] = json!(self.name());
        v
    }

    /// The path this op addresses, relative to the root (for the oracle).
    pub fn rel(&self) -> Option<&str> {
        match self {
            Op::Write { rel, .. }
            | Op::WriteIf { rel, .. }
            | Op::Read { rel, .. }
            | Op::Stat { rel }
            | Op::List { rel, .. }
            | Op::Search { rel, .. }
            | Op::Mkdir { rel } => Some(rel),
            Op::Exec { .. } => None,
        }
    }
}

fn sh(script: &str) -> Vec<String> {
    vec!["-c".into(), script.into()]
}

fn exec(command: &str, args: Vec<String>) -> Op {
    Op::Exec {
        command: command.into(),
        args,
        cwd: None,
        stdin: None,
        timeout_ms: None,
        stream: false,
    }
}

/// The tree every sequence starts from (the one of the `ShellFs` property test), made through `exec`.
pub fn seed() -> Vec<Op> {
    let files: &[(&str, &[u8])] = &[
        ("alpha.txt", b"alpha\nneedle one\nbeta\nNeedle two\n"),
        ("d/n.txt", b"foo bar\nneedle\r\nlast line no newline"),
        ("d/sub/x", b"x1\ne1\n\nneedle\n"),
        (".hidden", b"needle hidden\n"),
    ];
    let mut ops = vec![exec("mkdir", vec!["-p".into(), "--".into(), abs("d/sub")])];
    for (rel, data) in files {
        ops.push(env_write(rel, data.to_vec()));
    }
    ops
}

fn env_write(rel: &str, data: Vec<u8>) -> Op {
    Op::Exec {
        command: "sh".into(),
        args: vec!["-c".into(), "cat > \"$1\"".into(), "sh".into(), abs(rel)],
        cwd: None,
        stdin: Some(data),
        timeout_ms: None,
        stream: false,
    }
}

/// A name of 229 to 250 bytes is one whose write temp (`NAME.tmp.SUFFIX`) fits or not depending on
/// how long a suffix the client picks, which FILES-OVER-EXEC 5.3 leaves to it ("at least 64 bits"):
/// the Rust client's is 22 characters, so it refuses a name the others write. The spec decides
/// nothing there, so such a name is cut to 228 bytes, which every suffix fits.
fn unambiguous(name: String) -> String {
    if (229..=250).contains(&name.len()) {
        name[..228].to_string()
    } else {
        name
    }
}

/// Whether a symlink target (a path under `${ROOT}`, or a name relative to the link's directory)
/// stays in the root: no `..` component and nothing absolute outside `${ROOT}`.
fn stays_inside(target: &str) -> bool {
    let rest = target.strip_prefix("${ROOT}").unwrap_or(target);
    (target.starts_with("${ROOT}") || !target.starts_with('/'))
        && rest.split('/').all(|c| c != "..")
}

/// The paths a sequence has named so far, so later ops can address what earlier ones made.
struct Pool {
    known: Vec<String>,
    dirs: Vec<String>,
}

fn join(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

impl Pool {
    fn new() -> Self {
        let known: Vec<String> = ["alpha.txt", "d/n.txt", "d/sub/x", ".hidden", "d", "d/sub"]
            .map(String::from)
            .to_vec();
        Pool {
            known,
            dirs: vec!["d".into(), "d/sub".into()],
        }
    }

    fn remember(&mut self, p: &str) {
        if !self.known.iter().any(|k| k == p) {
            self.known.push(p.to_string());
        }
    }

    /// An existing path (6 in 10), else a new name under the root, a directory, a missing
    /// directory, or (now and then) a file.
    fn path(&mut self, tc: &TestCase) -> String {
        let p = if tc.draw(gs::integers::<u8>().max_value(9)) < 6 {
            tc.draw(gs::sampled_from(self.known.clone()))
        } else {
            let mut parents: Vec<String> = vec![String::new(); 4];
            parents.push("missing".into());
            for d in &self.dirs {
                parents.extend([d.clone(), d.clone()]);
            }
            if tc.draw(gs::integers::<u8>().max_value(4)) == 0 {
                let files: Vec<String> = self
                    .known
                    .iter()
                    .filter(|k| !self.dirs.contains(k))
                    .cloned()
                    .collect();
                if !files.is_empty() {
                    parents = files; // a file as the parent: the not-a-directory cases
                }
            }
            join(
                &tc.draw(gs::sampled_from(parents)),
                &unambiguous(fixtures::name(tc)),
            )
        };
        self.remember(&p);
        p
    }

    fn dir(&self, tc: &TestCase) -> String {
        let mut dirs = vec![String::new(); 2];
        dirs.extend(self.dirs.clone());
        dirs.push("missing".into());
        tc.draw(gs::sampled_from(dirs))
    }
}

/// A sequence of at most `max_steps` ops after the seed, then a read-back of everything named.
pub fn sequence(tc: &TestCase, max_steps: usize) -> Vec<Op> {
    let mut ops = seed();
    let mut pool = Pool::new();
    let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(max_steps));
    for _ in 0..n {
        match tc.draw(gs::integers::<u8>().max_value(28)) {
            0..=2 => {
                let rel = pool.path(tc);
                ops.push(Op::Write {
                    rel,
                    data: fixtures::content(tc, None, None),
                });
            }
            3 | 4 => {
                let rel = pool.path(tc);
                let data = fixtures::content(tc, None, None);
                let guard = match tc.draw(gs::integers::<u8>().max_value(3)) {
                    0 => Guard::Absent,
                    1 => Guard::Stale,
                    _ => {
                        ops.push(Op::Stat { rel: rel.clone() });
                        Guard::Current(ops.len() - 1)
                    }
                };
                ops.push(Op::WriteIf { rel, data, guard });
            }
            5..=7 => {
                let rel = pool.path(tc);
                let offset = tc.draw(gs::sampled_from(vec![0u64, 0, 1, 7, 40, 98304, 1 << 40]));
                let max = tc.draw(gs::sampled_from(vec![
                    None,
                    None,
                    Some(0u64),
                    Some(1),
                    Some(7),
                    Some(48 * 1024),
                    Some(98303),
                    Some(98304),
                    Some(98305),
                    Some(1 << 20),
                ]));
                ops.push(Op::Read { rel, offset, max });
            }
            8 | 9 => ops.push(Op::Stat { rel: pool.path(tc) }),
            10..=12 => {
                let rel = if tc.draw(gs::booleans()) {
                    pool.dir(tc)
                } else {
                    pool.path(tc)
                };
                ops.push(Op::List {
                    rel,
                    hidden: tc.draw(gs::booleans()),
                    limit: tc.draw(gs::sampled_from(vec![
                        None,
                        None,
                        Some(0),
                        Some(1),
                        Some(2),
                        Some(5),
                    ])),
                });
            }
            13..=15 => {
                let rel = if tc.draw(gs::integers::<u8>().max_value(7)) < 4 {
                    String::new()
                } else {
                    pool.path(tc)
                };
                let (pattern, literal, ignore_case) = fixtures::query(tc);
                let glob = match tc.draw(gs::integers::<u8>().max_value(5)) {
                    0 => Some(("*.txt".to_string(), false)),
                    1 => Some(("*.txt".to_string(), true)),
                    _ => None,
                };
                ops.push(Op::Search {
                    rel,
                    pattern,
                    literal,
                    ignore_case,
                    glob,
                    limit: tc.draw(gs::sampled_from(vec![0usize, 1, 2, 3, 100])),
                    before: tc.draw(gs::sampled_from(vec![0usize, 0, 1, 2, 150])),
                    after: tc.draw(gs::sampled_from(vec![0usize, 0, 1, 2, 150])),
                    raw: false,
                });
            }
            16 => {
                let rel = pool.path(tc);
                pool.dirs.push(rel.clone());
                ops.push(Op::Mkdir { rel });
            }
            // ---- the environment: changes made behind the client's back, through `exec` ----
            17 => {
                let rel = pool.path(tc);
                ops.push(env_write(&rel, fixtures::content(tc, None, None)));
            }
            18 => {
                let rel = pool.path(tc);
                pool.dirs.push(rel.clone());
                ops.push(exec("mkdir", vec!["-p".into(), "--".into(), abs(&rel)]));
            }
            19 => {
                let link = pool.path(tc);
                let target = match tc.draw(gs::integers::<u8>().max_value(4)) {
                    0 => "/dev/null".to_string(),
                    1 => abs("gone"),
                    2 => format!("{}.lnk", unambiguous(fixtures::name(tc))),
                    _ => abs(&pool.path(tc)),
                };
                // A link must never lead out of the root: `list` follows it, and the oracle and the
                // client would then read a live, shared directory at different moments. `/dev/null` is
                // a file, harmless; every other target is under `${ROOT}` or a bare name beside the link.
                if abs(&link) != target && (target == "/dev/null" || stays_inside(&target)) {
                    ops.push(exec(
                        "ln",
                        vec!["-s".into(), "--".into(), target, abs(&link)],
                    ));
                }
            }
            20 => {
                let rel = pool.path(tc);
                let mode = tc.draw(gs::sampled_from(vec![
                    "000", "444", "555", "600", "644", "700", "755", "111", "200",
                ]));
                ops.push(exec("chmod", vec![mode.into(), abs(&rel)]));
            }
            21 => {
                let rel = pool.path(tc);
                ops.push(exec(
                    "sh",
                    vec![
                        "-c".into(),
                        "chmod -R u+rwx \"$1\" 2>/dev/null; rm -rf -- \"$1\"".into(),
                        "sh".into(),
                        abs(&rel),
                    ],
                ));
            }
            // ---- layer 1: raw exec, the same answers expected of every client ----
            22 => ops.push(Op::Exec {
                command: "sh".into(),
                args: sh("printf out; printf err >&2; exit 3"),
                cwd: None,
                stdin: None,
                timeout_ms: None,
                stream: tc.draw(gs::booleans()),
            }),
            23 => ops.push(Op::Exec {
                command: "cat".into(),
                args: vec![],
                cwd: None,
                // Text: exec answers carry stdout as a string, so bytes that are not UTF-8 would not round-trip.
                stdin: Some(
                    tc.draw(gs::sampled_from(vec![
                        "",
                        "x",
                        "a\nb\n",
                        "漢字 ✓",
                        "tab\there",
                        "no newline",
                    ]))
                    .as_bytes()
                    .to_vec(),
                ),
                timeout_ms: None,
                stream: false, // exec_stream has no stdin field
            }),
            24 => ops.push(Op::Exec {
                command: "sh".into(),
                args: sh("ls -A | sort"),
                cwd: Some("${ROOT}".into()),
                stdin: None,
                timeout_ms: None,
                stream: false,
            }),
            25 => ops.push(Op::Exec {
                command: tc
                    .draw(gs::sampled_from(vec!["no-such-command-xyz", "/", "."]))
                    .into(),
                args: vec![],
                cwd: None,
                stdin: None,
                timeout_ms: None,
                stream: false,
            }),
            26 => ops.push(Op::Search {
                rel: String::new(),
                pattern: tc
                    .draw(gs::sampled_from(vec![
                        "(",
                        "[",
                        "*",
                        "a{",
                        "\\",
                        "(?i)needle",
                        "\\d+",
                        "needle|",
                        "^$",
                        "[[:alpha:]]+",
                        "x{2,1}",
                    ]))
                    .into(),
                literal: false,
                ignore_case: tc.draw(gs::booleans()),
                glob: None,
                limit: 5,
                before: 0,
                after: 0,
                raw: true,
            }),
            27 => ops.push(Op::Exec {
                command: "pwd".into(),
                args: vec![],
                cwd: Some(abs(&pool.path(tc))),
                stdin: None,
                timeout_ms: None,
                stream: tc.draw(gs::booleans()),
            }),
            _ => ops.push(Op::Exec {
                command: "sh".into(),
                args: sh("sleep 5"),
                cwd: None,
                stdin: None,
                timeout_ms: Some(300),
                stream: false,
            }),
        }
    }
    // The read-back: every named path is stat'ed and read, every directory listed, the tree searched.
    for p in pool.known.iter().take(20) {
        ops.push(Op::Stat { rel: p.clone() });
        ops.push(Op::Read {
            rel: p.clone(),
            offset: 0,
            max: None,
        });
        ops.push(Op::List {
            rel: p.clone(),
            hidden: true,
            limit: None,
        });
    }
    ops.push(Op::Search {
        rel: String::new(),
        pattern: "needle".into(),
        literal: true,
        ignore_case: true,
        glob: None,
        limit: 100,
        before: 1,
        after: 1,
        raw: false,
    });
    ops
}
