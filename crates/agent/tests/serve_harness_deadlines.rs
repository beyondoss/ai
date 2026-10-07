//! The `serve` suites' frame readers fail fast instead of hanging. A run that stalls — a scripted
//! reply taken by the wrong request, a question nobody will answer — used to leave a test blocked in
//! a read until the runner killed it, minutes later and with nothing said about where it stopped.
//! Every reader now has one shared deadline (`common::FRAME_DEADLINE`): stdout through
//! `common::child_frames`, WebSockets through `common::ws_next_frame`, and `skills_env::Serve`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A stdout that stays open and silent fails the read at the deadline.
#[test]
fn a_silent_stdout_fails_the_read_at_its_deadline() {
    let mut child = common::ChildGuard::spawn(
        Command::new("sleep")
            .arg("30")
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    );
    let mut frames = common::child_frames_within(&mut child, Duration::from_millis(300));
    let started = Instant::now();
    let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        common::read_until_response(&mut frames, "prompt")
    }));
    let message = *read.unwrap_err().downcast::<String>().unwrap();
    assert!(message.contains("stalled"), "{message}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
}

/// A WebSocket that stays open and silent fails the read at the deadline.
#[tokio::test]
async fn a_silent_websocket_fails_the_read_at_its_deadline() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        tokio::time::sleep(Duration::from_secs(30)).await;
        drop(ws);
    });
    let mut ws = common::ws_connect(port, None).await;
    let started = Instant::now();
    let read = tokio::spawn(async move {
        common::ws_next_frame_within(&mut ws, Duration::from_millis(300)).await
    })
    .await;
    let message = *read.unwrap_err().into_panic().downcast::<String>().unwrap();
    assert!(message.contains("stalled"), "{message}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    server.abort();
}

/// What may not appear in a scanned file: every way to reach a child's stdout pipe other than
/// `common::child_frames`. Matched on code with comments dropped and whitespace removed, so line
/// breaks and formatting cannot hide a use. `Output`'s `stdout` (captured bytes) and harness fields
/// that happen to be named `stdout` (a `Frames`) are not the pipe and are not flagged.
fn violations(source: &str) -> Vec<String> {
    let code: String = source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<String>()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let mut found = Vec::new();
    let mut flag = |what: &str, n: usize| {
        if n > 0 {
            found.push(format!("{what} x{n}"));
        }
    };
    // The pipe's type, and the guard's escape hatch for tests about the pipe itself.
    flag("`ChildStdout`", code.matches("ChildStdout").count());
    flag("`raw_stdout`", code.matches(".raw_stdout(").count());
    // Field access through `Option`'s methods on the child.
    for method in [
        "take", "as_mut", "as_ref", "unwrap", "expect", "replace", "insert",
    ] {
        let pattern = format!(".stdout.{method}(");
        flag(&format!("`{pattern}`"), code.matches(&pattern).count());
    }
    flag(
        "assignment to `.stdout`",
        count(&code, ".stdout=", |rest| !rest.starts_with('=')),
    );
    // The field taken by a free function: `std::mem::take(&mut child.stdout)`,
    // `Option::take(&mut child.stdout)`, `mem::replace(&mut child.stdout, ..)`, `mem::swap(..)`.
    for f in ["take(", "replace(", "swap("] {
        flag(
            &format!("`{f}&mut ….stdout`"),
            code.match_indices(&format!("{f}&mut"))
                .filter(|(i, m)| {
                    let arg = &code[i + m.len()..];
                    let end = arg.find([',', ')']).unwrap_or(arg.len());
                    arg[..end].ends_with(".stdout")
                })
                .count(),
        );
    }
    // A `Child` (or the guard) destructured with its `stdout` field.
    for ty in ["Child{", "ChildGuard{"] {
        flag(
            &format!("`{ty}..stdout..}}` pattern"),
            code.match_indices(ty)
                .filter(|(i, m)| {
                    let body = &code[i + m.len()..];
                    let end = body.find('}').unwrap_or(body.len());
                    body[..end].split([',', ':']).any(|f| f == "stdout")
                })
                .count(),
        );
    }
    // An unguarded spawn: a raw `Child`, whose `stdout` is still on it.
    flag("unguarded `.spawn()`", code.matches(".spawn()").count());
    found
}

fn count(code: &str, pattern: &str, keep: impl Fn(&str) -> bool) -> usize {
    code.match_indices(pattern)
        .filter(|(i, m)| keep(&code[i + m.len()..]))
        .count()
}

/// `tests/common/mod.rs` with the guard's own implementation — the one place that touches the
/// pipe — cut out: the `ChildGuard` struct and impl, and `child_frames_within`.
fn without_the_guard(source: &str) -> String {
    let mut out = source.to_string();
    for start in [
        "pub struct ChildGuard {",
        "impl ChildGuard {",
        "pub fn child_frames_within(",
    ] {
        let at = out
            .find(start)
            .unwrap_or_else(|| panic!("`{start}` is in tests/common/mod.rs"));
        let open = at + out[at..].find('{').unwrap();
        let mut depth = 0;
        let mut end = open;
        for (i, c) in out[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.replace_range(at..end, "");
    }
    out
}

/// Every bypass form is caught — on one line or spread over several, in any spelling.
#[test]
fn the_lint_catches_every_way_to_the_pipe() {
    for bypass in [
        "let out = child.stdout.take().unwrap();",
        "let out = child\n    .stdout\n    .take()\n    .unwrap();",
        "let out = std::mem::take(&mut child.stdout);",
        "let out = mem::take(\n    &mut child.stdout,\n);",
        "let out = Option::take(&mut child.stdout);",
        "let out = std::mem::replace(&mut child.stdout, None);",
        "std::mem::swap(&mut child.stdout, &mut mine);",
        "let out = child.stdout.as_mut().unwrap();",
        "child.stdout = None;",
        "let std::process::Child { stdout, .. } = raw;",
        "let Child { stdin, stdout: out, .. } = raw;",
        "let ChildGuard { stdout, .. } = guard;",
        "fn f(out: std::process::ChildStdout) {}",
        "let raw = Command::new(\"x\").stdout(Stdio::piped()).spawn().unwrap();",
        "let pipe = child.raw_stdout();",
    ] {
        assert!(!violations(bypass).is_empty(), "missed: {bypass:?}");
    }
}

/// What is not the pipe stays allowed: an `Output`'s captured bytes, a harness field named `stdout`
/// holding `Frames`, a `Command`'s stdout configuration, and anything in a comment.
#[test]
fn the_lint_leaves_what_is_not_the_pipe_alone() {
    for fine in [
        "let text = String::from_utf8_lossy(&output.stdout);",
        "assert!(output.stdout.is_empty());",
        "read_until(&mut s.stdout, \"x\", |f| true);",
        "read_until_response(&mut self.stdout, &name);",
        "cmd.stdout(Stdio::piped());",
        "let frames = common::child_frames(&mut child);",
        "let child = cmd.spawn_guarded();",
        "// child.stdout.take() would be wrong here",
        "let held = HeldPort::bind(); let child = held.spawn(&cmd, Stdio::null());",
    ] {
        assert_eq!(violations(fine), Vec::<String>::new(), "{fine:?}");
    }
}

/// And at run time: the guard holds the pipe, so every form reachable on a guarded child finds
/// nothing — only `child_frames` reads it.
#[test]
fn a_guarded_childs_stdout_field_is_empty_however_it_is_reached() {
    let mut child = common::ChildGuard::spawn(
        Command::new("sh")
            .args(["-c", "echo PIPED"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    );
    assert!(child.stdout.is_none());
    assert!(child.stdout.as_mut().is_none());
    assert!(child.stdout.take().is_none());
    assert!(std::mem::take(&mut child.stdout).is_none());
    assert!(Option::take(&mut child.stdout).is_none());
    let mut line = String::new();
    std::io::BufRead::read_line(&mut common::child_frames(&mut child), &mut line).unwrap();
    assert_eq!(line.trim(), "PIPED");
}

/// No test reaches a child's stdout pipe except through `common::child_frames`, in the `serve_*` and
/// `mcp_*` suites — top-level files and directory modules alike — and `tests/common`.
#[test]
fn only_child_frames_touches_a_childs_stdout() {
    let tests = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&tests).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let in_scope = name.starts_with("serve_") || name.starts_with("mcp_") || name == "common";
        if !in_scope || name == "serve_harness_deadlines.rs" {
            // This file spells out what it looks for.
            continue;
        }
        if path.is_dir() {
            rust_files(&path, &mut files);
        } else if name.ends_with(".rs") {
            files.push(path);
        }
    }
    assert!(
        files.iter().any(|f| f.ends_with("mcp_tasks_env/mod.rs")),
        "directory modules are scanned"
    );
    assert!(
        files.len() > 80,
        "the scan found the suites: {}",
        files.len()
    );
    let mut readers = 0;
    let mut found = Vec::new();
    for path in &files {
        let mut source = std::fs::read_to_string(path).unwrap();
        if path.ends_with("common/mod.rs") {
            source = without_the_guard(&source);
        }
        readers += source.matches("child_frames(").count();
        for v in violations(&source) {
            found.push(format!("{}: {v}", path.display()));
        }
    }
    assert!(
        found.is_empty(),
        "a child's stdout reached other than through common::child_frames:\n{}",
        found.join("\n")
    );
    assert!(
        readers > 250,
        "the suites read through child_frames ({readers})"
    );
}

/// No agent test picks a port, releases it, and hands it to a child: every child binds its own (port
/// 0, read back), is handed a socket the test holds (`HeldPort`), or is pointed at one the test holds
/// bound (`DeadPort`). The released-port helper is gone; this keeps the pattern from coming back.
#[test]
fn no_test_picks_a_port_and_releases_it() {
    let tests = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut files = Vec::new();
    let mut dirs = vec![tests];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs")
                && !path.ends_with("serve_harness_deadlines.rs")
            {
                files.push(path);
            }
        }
    }
    let found: Vec<String> = files
        .iter()
        .filter(|path| {
            let code: String = std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .map(|line| line.split("//").next().unwrap_or(""))
                .collect();
            code.contains("free_port(")
        })
        .map(|path| path.display().to_string())
        .collect();
    assert!(
        found.is_empty(),
        "a released port handed to a child: {found:?}"
    );
}
