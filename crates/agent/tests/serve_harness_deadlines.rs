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

/// Every place `source` binds a socket to port 0, reads the port the kernel gave it, and lets the
/// socket go without using it for anything else — the port then handed on for something else to bind,
/// which any process can take first. Recognised by what happens to the socket, not by a helper's
/// name:
///
/// - a **temporary**: `TcpListener::bind(…:0)…local_addr()` in one expression, the listener dropped
///   at the end of it (`.unwrap()`, `?`, `.and_then(|l| l.local_addr())` in between, any spelling);
/// - a **named** socket (`let l = TcpListener::bind(…:0)?` or `let s = TcpSocket::new_v4()?` then
///   `s.bind(…0…)`) whose only uses until the end of its block — or until the name is bound again —
///   are reading its address, configuring it, binding it, or `drop(l)`.
///
/// A socket that is also accepted on, listened on, converted, stored or passed anywhere is held, and
/// fine. Comments are dropped and whitespace normalised first ([`normalised`]), so formatting cannot
/// hide a case.
fn released_ports(source: &str) -> Vec<String> {
    let code = normalised(source);
    let mut found = Vec::new();
    // Temporaries: the bind's own statement reads the address.
    for ty in ["TcpListener::bind(", "TcpSocket::bind(", "UdpSocket::bind("] {
        for (at, _) in code.match_indices(ty) {
            let stmt = &code[at..];
            let stmt = &stmt[..stmt.find(';').unwrap_or(stmt.len())];
            let close = matching_paren(stmt, stmt.find('(').unwrap());
            if binds_port_zero(&stmt[..close]) && stmt[close..].contains(".local_addr()") {
                found.push(format!("a temporary listener's port ({})", excerpt(stmt)));
            }
        }
    }
    // Named sockets: follow the name to the end of its scope.
    for (at, _) in code.match_indices("let ") {
        if code[..at]
            .chars()
            .last()
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
        {
            continue;
        }
        let after = &code[at + 4..];
        let after = after.strip_prefix("mut ").unwrap_or(after);
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let Some(eq) = after.find('=') else { continue };
        if name.is_empty()
            || !after[name.len()..eq]
                .chars()
                .all(|c| c.is_alphanumeric() || "<>_:& ".contains(c))
        {
            continue;
        }
        let stmt_end = after.find(';').unwrap_or(after.len());
        let init = &after[eq + 1..stmt_end];
        let rest = scope_after(&after[stmt_end..], &name);
        let bound_zero_here = (init.contains("TcpListener::bind(")
            || init.contains("UdpSocket::bind("))
            && binds_port_zero(init);
        let socket_here =
            init.contains("TcpSocket::new_v4(") || init.contains("TcpSocket::new_v6(");
        if !bound_zero_here && !socket_here {
            continue;
        }
        let uses = uses_of(rest, &name);
        let reads_port = uses.iter().any(|u| u.starts_with(".local_addr()"));
        let bound_zero = bound_zero_here
            || uses
                .iter()
                .any(|u| u.starts_with(".bind(") && binds_port_zero(&u[..matching_paren(u, 5)]));
        let let_go = uses.iter().all(|u| {
            u.starts_with(".local_addr()")
                || u.starts_with(".bind(")
                || u.starts_with(".set_")
                || *u == "drop"
        });
        if bound_zero && reads_port && let_go {
            found.push(format!(
                "`{name}`: bound to port 0, its port read, then let go"
            ));
        }
    }
    found
}

/// The index just past the `)` matching the `(` at `open`.
fn matching_paren(s: &str, open: usize) -> usize {
    let mut depth = 0;
    for (i, c) in s[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return open + i + 1;
                }
            }
            _ => {}
        }
    }
    s.len()
}

/// Whether a bind's arguments ask for port 0.
fn binds_port_zero(args: &str) -> bool {
    args.contains(":0\"") || args.contains(",0)") || args.contains(",0))")
}

/// `source` without comments, and with whitespace removed except a single space between two
/// identifier characters (`for x in listener.incoming()` keeps `in listener` apart), so a pattern
/// matches however the code is laid out.
fn normalised(source: &str) -> String {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let code: String = source
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let mut out = String::with_capacity(code.len());
    let mut pending_space = false;
    for c in code.chars() {
        if c.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space && out.chars().last().is_some_and(ident) && ident(c) {
            out.push(' ');
        }
        pending_space = false;
        out.push(c);
    }
    out
}

/// The code from a binding to the end of its block, or through the statement that binds `name` again
/// (whose right-hand side may still use the old binding: `let l = from_std(l)`).
fn scope_after<'a>(code: &'a str, name: &str) -> &'a str {
    let mut depth = 0i32;
    for (i, c) in code.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return &code[..i];
                }
            }
            _ => {}
        }
        let rebinds = [
            format!("let {name}="),
            format!("let mut {name}="),
            format!("let {name}:"),
        ];
        if rebinds.iter().any(|r| code[i..].starts_with(r.as_str())) {
            let end = code[i..].find(';').map_or(code.len(), |e| i + e);
            return &code[..end];
        }
    }
    code
}

/// Every use of identifier `name` in `code`, as the text from the identifier's end — or, for a
/// `drop(name)`, the text `drop`.
fn uses_of<'a>(code: &'a str, name: &str) -> Vec<&'a str> {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    code.match_indices(name)
        .filter(|(i, _)| !code[..*i].chars().last().is_some_and(ident))
        .filter(|(i, m)| !code[i + m.len()..].chars().next().is_some_and(ident))
        .filter(|(i, _)| !code[..*i].ends_with('.'))
        .map(|(i, m)| {
            if code[..i].ends_with("drop(") && code[i + m.len()..].starts_with(')') {
                "drop"
            } else {
                &code[i + m.len()..]
            }
        })
        .collect()
}

fn excerpt(s: &str) -> &str {
    &s[..s.len().min(60)]
}

/// The lint recognises the shape — a port-0 socket whose port is read and which is then let go —
/// however it is written, and leaves a socket that is held, served or passed on alone.
#[test]
fn the_released_port_lint_catches_what_it_should() {
    for bad in [
        // A temporary.
        "let p = TcpListener::bind(\"127.0.0.1:0\").unwrap().local_addr().unwrap().port();",
        "let p = std::net::TcpListener::bind(\"127.0.0.1:0\")\n    .and_then(|l| l.local_addr())\n    .unwrap()\n    .port();",
        "let p = TcpListener::bind((\"127.0.0.1\", 0)).await?.local_addr()?.port();",
        // Named, under any name, in a helper or inline.
        "fn pick() -> u16 { let l = TcpListener::bind(\"127.0.0.1:0\").unwrap(); l.local_addr().unwrap().port() }",
        "let sock = std::net::TcpListener::bind(\"127.0.0.1:0\")?;\nlet port = sock.local_addr()?.port();\ndrop(sock);\nspawn_child(port);",
        "let reserve = TcpListener::bind(\"0.0.0.0:0\").unwrap();\nlet n = reserve.local_addr().unwrap().port();\n}\nchild(n);",
        // A socket bound to port 0 after creation.
        "let s = TcpSocket::new_v4()?;\ns.set_reuseaddr(true)?;\ns.bind(\"127.0.0.1:0\".parse().unwrap())?;\nlet p = s.local_addr()?.port();\ndrop(s);",
    ] {
        assert!(!released_ports(bad).is_empty(), "missed: {bad:?}");
    }
    for fine in [
        // Held and served.
        "let listener = TcpListener::bind(\"127.0.0.1:0\").await.unwrap();\nlet port = listener.local_addr().unwrap().port();\ntokio::spawn(serve(listener));",
        // Held in a struct for the test's life.
        "let socket = TcpSocket::new_v4().unwrap();\nsocket.bind(addr0).unwrap();\nlet port = socket.local_addr().unwrap().port();\nSelf { _socket: socket, port }",
        "let l = TcpListener::bind(\"127.0.0.1:0\").unwrap();\nlet port = l.local_addr().unwrap().port();\nlet (conn, _) = l.accept().unwrap();",
        // A fixed port: nothing picked.
        "let l = TcpListener::bind(\"127.0.0.1:8080\").unwrap();\nlet a = l.local_addr().unwrap();\ndrop(l);",
        "// let p = TcpListener::bind(\"127.0.0.1:0\").unwrap().local_addr().unwrap().port();",
        "let dead = DeadPort::bind();",
        // Used right after a keyword, which a whitespace-blind match would glue to it.
        "let listener = TcpListener::bind(\"127.0.0.1:0\").unwrap();\nlet addr = listener.local_addr().unwrap();\nthread::spawn(move || { for s in listener.incoming() { drop(s); } });",
        // Re-bound by a statement that consumes the old binding.
        "let listener = std::net::TcpListener::bind(\"127.0.0.1:0\").unwrap();\nlet port = listener.local_addr().unwrap().port();\nlet listener = tokio::net::TcpListener::from_std(listener).unwrap();",
    ] {
        assert_eq!(released_ports(fine), Vec::<String>::new(), "{fine:?}");
    }
}

/// No code in the workspace — any crate's sources, tests or benches — picks a port, releases it and
/// hands it to something else to bind: every child binds its own (port 0, read back: `serve`'s
/// announcement, the gateway's own `LISTEN` sockets, `nats-server --ports_file_dir`), is handed a
/// socket the test holds (`HeldPort`, the forwarder in front of a restartable `nats-server`), or is
/// pointed at one held bound (`DeadPort`). The released-port helpers are gone from every crate; this
/// keeps the pattern from coming back.
#[test]
fn no_test_picks_a_port_and_releases_it() {
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let mut files = Vec::new();
    let mut dirs = vec![crates.clone()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if path.is_dir() {
                if name != "target" && !name.starts_with('.') {
                    dirs.push(path);
                }
            } else if name.ends_with(".rs") && name != "serve_harness_deadlines.rs" {
                files.push(path);
            }
        }
    }
    for krate in ["agent", "gateway", "verify", "fleet-sim", "test-support"] {
        assert!(
            files.iter().any(|f| f.starts_with(crates.join(krate))),
            "the scan covers the {krate} crate"
        );
    }
    let found: Vec<String> = files
        .iter()
        .flat_map(|path| {
            let source = std::fs::read_to_string(path).unwrap_or_default();
            released_ports(&source)
                .into_iter()
                .map(move |what| format!("{}: {what}", path.display()))
        })
        .collect();
    assert!(
        found.is_empty(),
        "a port picked free and released for something else to bind:\n{}",
        found.join("\n")
    );
}
