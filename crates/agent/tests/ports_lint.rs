//! Port 0 is spelled in one module: `beyond_ai_test_support::ports`.
//!
//! Binding port 0 and then reading the port the kernel picked is how code gets a port no other
//! process holds. Releasing that socket and passing the number on for something else to bind
//! reopens the race: any process can take the port in between. A lint that tried to tell a held
//! socket from a released one had to follow the socket through bindings, wrappers, closures and
//! helpers in other files, and it kept missing new shapes. This rule needs none of that, so a new
//! shape cannot slip past it. **A zero port is not written anywhere in the workspace except
//! `crates/test-support/src/ports.rs`**, the one module allowed to bind and read a port. Every test
//! that needs a port uses its helpers:
//!
//! - a listener held for the test's life: `ports::listener()`, `ports::tokio_listener()`;
//! - a bound socket that never listens: `ports::bound_socket()`, `DeadPort`;
//! - a child that binds port 0 itself and reports what it got (`common::spawn_listening`, the
//!   gateway's `GATEWAY_LISTENERS`, `nats-server --ports_file_dir`).
//!
//! Some code outside `ports.rs` binds port 0 and keeps the listener for real: the MCP OAuth
//! callback, the fixture servers that announce their port, and the command lines that ask a child
//! to bind port 0 and report it. Each such site has a `// port-0: <why it is held>` comment in place
//! and is counted in [`ALLOWED`], so a new one shows up as a change to this file.
//!
//! What counts as writing a zero port:
//!
//! - an address literal with port 0: `"127.0.0.1:0"`, `"[::1]:0"`, `"localhost:0"`,
//!   `"{host}:0"`, `":0"`;
//! - a zero (`0`, `0u16`, `0_u16`, `u16::MIN`, `0 as u16`, `Default::default()`, or a `const`,
//!   `static` or `let` bound to one) used as the port of:
//!   - a `bind` call;
//!   - `SocketAddr::new`, `SocketAddrV4::new`, or `SocketAddrV6::new` (the second argument);
//!   - `set_port`;
//!   - an address tuple whose first element is an IP (`("127.0.0.1", 0)`, `([127, 0, 0, 1], 0)`,
//!     `(Ipv4Addr::LOCALHOST, 0)`, `(ip, 0)`);
//! - a zero formatted in as an address's port: `format!("127.0.0.1:{}", 0)`,
//!   `format!("{}:{}", host, 0)`, `format!("{host}:{port}", port = 0)`, or `{port}` capturing a
//!   `let port = 0`;
//! - a zero in a port field: `Config { port: 0, .. }`, `listen_port: 0`.
//!
//! Comments are skipped, and string literals are only checked for addresses. Whitespace is
//! ignored, so layout cannot hide a case.
//!
//! **What a text rule cannot see.** A zero that only becomes a port later, through a value the rule
//! cannot follow, passes:
//!
//! - an environment default: `env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(0)`,
//!   later passed to a bind;
//! - a string default: `unwrap_or_else(|| "0".into())`, later formatted into an address;
//! - a port computed at run time: `n - n`.
//!
//! Catching these needs the value's flow, which is exactly the analysis this rule gave up. The first
//! two are plausible by accident, so review still matters where a port comes from configuration; the
//! last would be deliberate evasion.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Every file outside `ports.rs` that binds port 0 and keeps the socket, with how many annotated
/// (`// port-0:`) sites it has. Adding one is a reviewed change here, not just a comment there.
const ALLOWED: &[(&str, usize)] = &[
    // `agent mcp login`: the callback listener is held and the redirect URI names its port.
    ("agent/src/main.rs", 1),
    // The `CallbackServer` under test holds its listener.
    ("agent/src/oauth/callback_server.rs", 1),
    // Fixture servers, run as children: each binds port 0, keeps it, and announces it.
    ("agent/src/bin/mcp_fixture_events_server.rs", 1),
    ("agent/src/bin/mcp_fixture_tasks_server.rs", 1),
    ("agent/src/bin/mcp_skills_fixture_server.rs", 1),
    // Not a bind: the resolver's lookup takes a port, and only the addresses it returns are used.
    ("agent/src/tools/web/ssrf.rs", 1),
    // Command lines that ask a child `serve` to bind port 0 and report the port it got.
    ("agent/tests/common/mod.rs", 1),
    ("agent/tests/serve_service_mode.rs", 2),
    ("agent/benches/serve_runtime.rs", 1),
    ("fleet-sim/src/replica.rs", 2),
];

/// A source file with comments blanked out (newlines kept) and its string literals located.
struct Lexed {
    /// The source's characters, with every comment character replaced by a space.
    code: Vec<char>,
    /// Each string literal as (start, end) indices of its contents in `code`.
    literals: Vec<(usize, usize)>,
}

fn lex(source: &str) -> Lexed {
    let s: Vec<char> = source.chars().collect();
    let mut code = s.clone();
    let mut literals = Vec::new();
    let mut i = 0;
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    while i < s.len() {
        let c = s[i];
        let next = s.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < s.len() && s[i] != '\n' {
                code[i] = ' ';
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            let mut depth = 0;
            while i < s.len() {
                if s[i] == '/' && s.get(i + 1) == Some(&'*') {
                    depth += 1;
                    code[i] = ' ';
                    code[i + 1] = ' ';
                    i += 2;
                } else if s[i] == '*' && s.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    code[i] = ' ';
                    code[i + 1] = ' ';
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    if s[i] != '\n' {
                        code[i] = ' ';
                    }
                    i += 1;
                }
            }
        } else if (c == 'r' || (c == 'b' && next == Some('r'))) && !(i > 0 && ident(s[i - 1])) && {
            let mut j = i + if c == 'b' { 2 } else { 1 };
            while s.get(j) == Some(&'#') {
                j += 1;
            }
            s.get(j) == Some(&'"')
        } {
            // A raw string: `r#"…"#`, closed by a quote and the same number of hashes.
            let mut j = i + if c == 'b' { 2 } else { 1 };
            let mut hashes = 0;
            while s[j] == '#' {
                hashes += 1;
                j += 1;
            }
            let start = j + 1;
            let mut k = start;
            while k < s.len() {
                if s[k] == '"' && (0..hashes).all(|h| s.get(k + 1 + h) == Some(&'#')) {
                    break;
                }
                k += 1;
            }
            literals.push((start, k.min(s.len())));
            i = k + 1 + hashes;
        } else if c == '"' {
            let start = i + 1;
            let mut k = start;
            while k < s.len() && s[k] != '"' {
                if s[k] == '\\' {
                    k += 1;
                }
                k += 1;
            }
            literals.push((start, k.min(s.len())));
            i = k + 1;
        } else if c == '\'' {
            // A char literal (`'"'`, `'\''`), or a lifetime (`'a`), which has no closing quote.
            if next == Some('\\') {
                // Past the escaped character, which may itself be a quote (`'\''`).
                let mut k = i + 3;
                while k < s.len() && s[k] != '\'' {
                    k += 1;
                }
                i = k + 1;
            } else if s.get(i + 2) == Some(&'\'') {
                i += 3;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    Lexed { code, literals }
}

/// Whether `text` (a literal's contents) holds an address with port 0.
fn literal_has_port_zero(text: &[char]) -> bool {
    let t: String = text.iter().collect();
    t.match_indices(":0").any(|(at, _)| {
        let after = t[at + 2..].chars().next();
        if after.is_some_and(|c| c.is_ascii_alphanumeric() || c == '.' || c == '/' || c == ':') {
            return false;
        }
        names_a_host(&t[..at])
    })
}

/// The placeholders of a format string that stand for an address's port (`"127.0.0.1:{}"`,
/// `"{host}:{port}"`), each as the argument it takes: `Ok(i)` the `i`th positional argument, or
/// `Err(name)` a named one (or a captured variable).
fn port_placeholders(t: &str) -> Vec<Result<usize, String>> {
    let mut out = Vec::new();
    let mut positional = 0;
    let mut i = 0;
    while let Some(open) = t[i..].find('{').map(|o| o + i) {
        if t[open + 1..].starts_with('{') {
            i = open + 2;
            continue;
        }
        let Some(close) = t[open..].find('}').map(|c| c + open) else {
            break;
        };
        let inner = &t[open + 1..close];
        let arg = inner.split(':').next().unwrap_or("");
        let which = if arg.is_empty() {
            positional += 1;
            Ok(positional - 1)
        } else if let Ok(n) = arg.parse::<usize>() {
            Ok(n)
        } else {
            Err(arg.to_string())
        };
        if open > 0 && t[..open].ends_with(':') && names_a_host(&t[..open - 1]) {
            out.push(which);
        }
        i = close + 1;
    }
    out
}

/// Whether `before` (the text up to an address's `:`) ends with a host: a dotted quad, `]`,
/// `localhost`, a `{placeholder}`, or nothing at all.
fn names_a_host(before: &str) -> bool {
    {
        if before.is_empty() || before.ends_with(']') || before.ends_with('}') {
            return true;
        }
        if before.ends_with("localhost") {
            return true;
        }
        // A dotted quad right before the colon.
        let quad: String = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let parts: Vec<&str> = quad.split('.').collect();
        parts.len() == 4
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.parse::<u8>().is_ok())
    }
}

/// The code with whitespace removed (a single space kept between two identifier characters), each
/// character with its index in the lexed source and whether it lies inside a string literal.
fn normalised(lexed: &Lexed) -> Vec<(char, usize, bool)> {
    let mut in_literal = vec![false; lexed.code.len()];
    for &(a, b) in &lexed.literals {
        // The quotes too, so `"…"` is one unit.
        for flag in &mut in_literal[a.saturating_sub(1)..(b + 1).min(lexed.code.len())] {
            *flag = true;
        }
    }
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut out: Vec<(char, usize, bool)> = Vec::new();
    let mut pending_space = false;
    for (i, &c) in lexed.code.iter().enumerate() {
        if c.is_whitespace() && !in_literal[i] {
            pending_space = true;
            continue;
        }
        if pending_space
            && ident(c)
            && out.last().is_some_and(|&(p, _, lit)| ident(p) && !lit)
            && !in_literal[i]
        {
            out.push((' ', i, false));
        }
        pending_space = false;
        out.push((c, i, in_literal[i]));
    }
    out
}

const ZERO_SPELLINGS: &[&str] = &[
    "0u16",
    "0_u16",
    "u16::MIN",
    "0 as u16",
    "Default::default()",
    "u16::default()",
    "0",
];

/// Names bound to a zero: `const`/`static` ones (`const ANY: u16 = 0;`) are returned in `global`,
/// `let` ones in `local`.
fn zero_names(text: &str, global: &mut HashSet<String>, local: &mut HashSet<String>) {
    for kw in ["const ", "static ", "let mut ", "let "] {
        for (at, _) in text.match_indices(kw) {
            if text[..at]
                .chars()
                .last()
                .is_some_and(|c| c.is_alphanumeric() || c == '_')
            {
                continue;
            }
            let rest = &text[at + kw.len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            let mut after = &rest[name.len()..];
            if let Some(typed) = after.strip_prefix(':') {
                let ty: String = typed
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                after = &typed[ty.len()..];
            }
            let Some(value) = after.strip_prefix('=') else {
                continue;
            };
            if ZERO_SPELLINGS
                .iter()
                .any(|z| value.strip_prefix(z).is_some_and(|r| r.starts_with(';')))
            {
                if kw.starts_with("let") {
                    local.insert(name);
                } else {
                    global.insert(name);
                }
            }
        }
    }
}

/// Whether an address tuple's first element names an IP.
fn ip_like(first: &str) -> bool {
    let first = first.trim_start_matches('&');
    if let Some(lit) = first.strip_prefix('"') {
        let lit = lit.trim_end_matches('"');
        let parts: Vec<&str> = lit.split('.').collect();
        return lit == "localhost"
            || lit.contains("::")
            || (parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok()));
    }
    if first.starts_with('[') && first[1..].starts_with(|c: char| c.is_ascii_digit()) {
        return true;
    }
    if ["Ipv4Addr", "Ipv6Addr", "IpAddr", "LOCALHOST", "UNSPECIFIED"]
        .iter()
        .any(|k| first.contains(k))
    {
        return true;
    }
    first
        .split(['.', ':'])
        .map(|s| s.trim_end_matches("()").to_ascii_lowercase())
        .any(|s| {
            ["ip", "host", "addr", "loopback", "localhost"].contains(&s.as_str())
                || s.ends_with("_ip")
                || s.ends_with("_host")
        })
}

/// The top-level, comma-separated elements of the group opened at `open`, and where it closes.
fn elements(n: &[(char, usize, bool)], open: usize) -> (Vec<String>, usize) {
    let mut depth = 0i32;
    let mut out = vec![String::new()];
    for (i, &(c, _, lit)) in n.iter().enumerate().skip(open + 1) {
        if !lit {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => {
                    if depth == 0 {
                        // A trailing comma (`(ip, 0,)`, as rustfmt lays out a long tuple).
                        if out.len() > 1 && out.last().is_some_and(String::is_empty) {
                            out.pop();
                        }
                        return (out, i);
                    }
                    depth -= 1;
                }
                ',' if depth == 0 => {
                    out.push(String::new());
                    continue;
                }
                _ => {}
            }
        }
        out.last_mut().unwrap().push(c);
    }
    (out, n.len())
}

/// The path called right before `open` (`TcpListener::bind`, `SocketAddr::new`), or empty for a
/// bare tuple.
fn callee(n: &[(char, usize, bool)], open: usize) -> String {
    let mut s: Vec<char> = n[..open]
        .iter()
        .rev()
        .take_while(|&&(c, _, lit)| {
            !lit && (c.is_alphanumeric() || c == '_' || c == ':' || c == '.')
        })
        .map(|&(c, ..)| c)
        .collect();
    s.reverse();
    s.into_iter().collect()
}

/// Each place in `source` that writes a zero port, as the index in `source`'s characters.
fn port_zero_sites(source: &str, global_zero: &HashSet<String>) -> Vec<usize> {
    let lexed = lex(source);
    let mut sites = Vec::new();
    for &(a, b) in &lexed.literals {
        if literal_has_port_zero(&lexed.code[a..b]) {
            sites.push(a);
        }
    }
    let n = normalised(&lexed);
    let text: String = n.iter().map(|&(c, ..)| c).collect();
    let mut local_zero = HashSet::new();
    let mut ignored = HashSet::new();
    zero_names(&text, &mut ignored, &mut local_zero);
    let is_zero =
        |e: &str| ZERO_SPELLINGS.contains(&e) || global_zero.contains(e) || local_zero.contains(e);
    for open in (0..n.len()).filter(|&i| n[i].0 == '(' && !n[i].2) {
        let (els, close) = elements(&n, open);
        let called = callee(&n, open);
        let zero_at = |k: usize| els.get(k).is_some_and(|e| is_zero(e));
        let site = n.get(close.saturating_sub(1)).map_or(0, |&(_, at, _)| at);
        let macro_call = open > 0 && n[open - 1].0 == '!' && !n[open - 1].2;
        let hit = if macro_call {
            // `format!("127.0.0.1:{}", 0)`: an address format string whose port placeholder takes
            // a zero, positional, named (`port = 0`) or captured (`{port}` with `let port = 0`).
            let lit = els
                .first()
                .map_or("", |f| f.trim_start_matches('r').trim_matches('#'));
            let args = els.get(1..).unwrap_or_default();
            let positional: Vec<&String> = args.iter().filter(|e| !e.contains('=')).collect();
            lit.starts_with('"')
                && port_placeholders(lit.trim_matches('"'))
                    .iter()
                    .any(|which| match which {
                        Ok(i) => positional.get(*i).is_some_and(|e| is_zero(e)),
                        // A captured variable counts only when it is named as a port: a
                        // `{seed}:{attempt}` key whose counter starts at 0 is not an address.
                        Err(name) => {
                            ((name == "port" || name.ends_with("_port")) && is_zero(name))
                                || args.iter().any(|e| {
                                    e.split_once('=')
                                        .is_some_and(|(k, v)| k == name && is_zero(v))
                                })
                        }
                    })
        } else if called.ends_with("SocketAddrV6::new") {
            zero_at(1)
        } else if called.ends_with("set_port") {
            els.len() == 1 && zero_at(0)
        } else if els.len() == 2 && zero_at(1) {
            called.ends_with("bind")
                || called.ends_with("SocketAddr::new")
                || called.ends_with("SocketAddrV4::new")
                || ip_like(&els[0])
        } else {
            false
        };
        if hit {
            sites.push(site);
        }
    }
    // A port field set to zero: `Config { port: 0, .. }`, `listen_port: 0`.
    for (at, _) in text.match_indices(':') {
        let name: String = text[..at]
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let opens_field = text[..at - name.len()].ends_with(['{', ',']);
        let value = &text[at + 1..];
        let zero = ZERO_SPELLINGS.iter().any(|z| {
            value
                .strip_prefix(z)
                .is_some_and(|r| r.starts_with([',', '}']))
        });
        if (name == "port" || name.ends_with("_port"))
            && opens_field
            && zero
            && !n[at].2
            && !text[at + 1..].starts_with(':')
            && !text[..at].ends_with(':')
        {
            sites.push(n[at].1);
        }
    }
    sites
}

/// Whether the line holding character `at`, or the comment lines right above it, carry a
/// `port-0:` annotation.
fn annotated(source: &str, at: usize) -> bool {
    let lines: Vec<&str> = source.lines().collect();
    let line = source.chars().take(at).filter(|&c| c == '\n').count();
    if lines.get(line).is_some_and(|l| l.contains("port-0:")) {
        return true;
    }
    lines[..line]
        .iter()
        .rev()
        .take_while(|l| l.trim_start().starts_with("//"))
        .any(|l| l.contains("port-0:"))
}

fn line_of(source: &str, at: usize) -> (usize, String) {
    let line = source.chars().take(at).filter(|&c| c == '\n').count();
    (
        line + 1,
        source.lines().nth(line).unwrap_or("").trim().to_string(),
    )
}

/// The unannotated zero-port sites in one source, for the pinned cases.
fn findings(source: &str) -> Vec<String> {
    let lexed = lex(source);
    let n = normalised(&lexed);
    let text: String = n.iter().map(|&(c, ..)| c).collect();
    let mut global = HashSet::new();
    zero_names(&text, &mut global, &mut HashSet::new());
    port_zero_sites(source, &global)
        .into_iter()
        .filter(|&at| !annotated(source, at))
        .map(|at| line_of(source, at).1)
        .collect()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rust_files(&path, out);
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

#[test]
fn port_zero_is_written_only_in_the_ports_module() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut files = Vec::new();
    rust_files(crates, &mut files);
    for krate in [
        "agent",
        "agent-core",
        "gateway",
        "verify",
        "fleet-sim",
        "test-support",
    ] {
        assert!(
            files.iter().any(|f| f.starts_with(crates.join(krate))),
            "the scan covers the {krate} crate"
        );
    }
    let this = Path::new(file!()).file_name().unwrap();
    let ports = crates.join("test-support/src/ports.rs");
    assert!(files.contains(&ports), "the ports module is where it was");
    let files: Vec<PathBuf> = files
        .into_iter()
        .filter(|f| *f != ports && f.file_name() != Some(this))
        .collect();
    let sources: Vec<String> = files
        .iter()
        .map(|f| std::fs::read_to_string(f).unwrap())
        .collect();
    // `const ANY: u16 = 0` in one file is a zero port when another file uses it.
    let mut global = HashSet::new();
    for s in &sources {
        let n = normalised(&lex(s));
        let text: String = n.iter().map(|&(c, ..)| c).collect();
        zero_names(&text, &mut global, &mut HashSet::new());
    }
    let mut unannotated = Vec::new();
    let mut held: Vec<(String, usize)> = Vec::new();
    for (file, source) in files.iter().zip(&sources) {
        let rel = file
            .strip_prefix(crates)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let mut count = 0;
        for at in port_zero_sites(source, &global) {
            if annotated(source, at) {
                count += 1;
            } else {
                let (line, text) = line_of(source, at);
                unannotated.push(format!("{rel}:{line}: {text}"));
            }
        }
        if count > 0 {
            held.push((rel, count));
        }
    }
    unannotated.sort();
    unannotated.dedup();
    assert!(
        unannotated.is_empty(),
        "port 0 written outside beyond_ai_test_support::ports — use its helpers (`listener()`, \
         `tokio_listener()`, `bound_socket()`, `DeadPort`, or a child that reports its port):\n{}",
        unannotated.join("\n")
    );
    held.sort();
    let mut allowed: Vec<(String, usize)> =
        ALLOWED.iter().map(|&(f, n)| (f.to_string(), n)).collect();
    allowed.sort();
    assert_eq!(
        held, allowed,
        "the annotated (`// port-0:`) sites differ from ALLOWED: update it in the same change"
    );
}

/// Every way of releasing a picked port that the old data-flow lint had to chase is now caught,
/// because each one writes a zero port. So are listeners that are held, if they sit outside
/// `ports.rs` without an annotation.
#[test]
fn the_rule_catches_every_spelling_of_port_zero() {
    for bad in [
        // A temporary.
        "let p = TcpListener::bind(\"127.0.0.1:0\").unwrap().local_addr().unwrap().port();",
        "let p = std::net::TcpListener::bind(\"127.0.0.1:0\")\n    .and_then(|l| l.local_addr())\n    .unwrap()\n    .port();",
        "let p = TcpListener::bind((\"127.0.0.1\", 0)).await?.local_addr()?.port();",
        // Named, in a helper or inline.
        "fn pick() -> u16 { let l = TcpListener::bind(\"127.0.0.1:0\").unwrap(); l.local_addr().unwrap().port() }",
        "let sock = std::net::TcpListener::bind(\"127.0.0.1:0\")?;\nlet port = sock.local_addr()?.port();\ndrop(sock);\nspawn_child(port);",
        "let reserve = TcpListener::bind(\"0.0.0.0:0\").unwrap();\nlet n = reserve.local_addr().unwrap().port();\n}\nchild(n);",
        // A socket bound after creation.
        "let s = TcpSocket::new_v4()?;\ns.set_reuseaddr(true)?;\ns.bind(\"127.0.0.1:0\".parse().unwrap())?;\nlet p = s.local_addr()?.port();\ndrop(s);",
        // A port-0 address held in a variable or const.
        "let a = \"127.0.0.1:0\";\nlet l = TcpListener::bind(a).unwrap();\nlet p = l.local_addr().unwrap().port();\ndrop(l);",
        "const ANY: &str = \"127.0.0.1:0\";\nfn pick() -> u16 { TcpListener::bind(ANY).unwrap().local_addr().unwrap().port() }",
        // A typed zero.
        "let p = TcpListener::bind((\"127.0.0.1\", 0u16)).unwrap().local_addr().unwrap().port();",
        "let p = TcpListener::bind((Ipv4Addr::LOCALHOST, 0_u16)).unwrap().local_addr().unwrap().port();",
        "let p = TcpListener::bind((Ipv4Addr::LOCALHOST, u16::MIN)).unwrap();",
        "let p = TcpListener::bind((ip, 0 as u16)).unwrap();",
        // The socket handed to a helper that only reads its address.
        "fn port_of(l: &TcpListener) -> u16 { l.local_addr().unwrap().port() }\nfn pick() -> u16 { let l = TcpListener::bind(\"127.0.0.1:0\").unwrap(); port_of(&l) }",
        // A tuple pattern, `let _ =`, a self-rebind.
        "let (l, _) = (TcpListener::bind(\"127.0.0.1:0\").unwrap(), 1);\nlet p = l.local_addr().unwrap().port();\ndrop(l);",
        "let l = TcpListener::bind(\"127.0.0.1:0\").unwrap();\nlet p = l.local_addr().unwrap().port();\nlet _ = l;",
        "let l = TcpListener::bind(\"127.0.0.1:0\").unwrap();\nlet l = l;\nlet p = l.local_addr().unwrap().port();\ndrop(l);",
        // A closure that only reads the address.
        "let port_of = |l: &TcpListener| l.local_addr().unwrap().port();\nlet l = TcpListener::bind(\"127.0.0.1:0\").unwrap();\nlet p = port_of(&l);\ndrop(l);",
        // A listener held in an `Option`.
        "let l = Some(TcpListener::bind(\"127.0.0.1:0\").unwrap());\nlet p = l.as_ref().unwrap().local_addr().unwrap().port();\ndrop(l);",
        "let l = Some(TcpListener::bind(\"127.0.0.1:0\").unwrap());\nlet p = l.as_ref().expect(\"bound\").local_addr().unwrap().port();",
        // A zero port in a const or a let.
        "const ANY: u16 = 0;\nlet p = TcpListener::bind((\"127.0.0.1\", ANY)).unwrap().local_addr().unwrap().port();",
        "let mut port = 0;\nlet l = TcpListener::bind((host, port)).unwrap();",
        // A helper in another file: the bind is still written here.
        "let l = TcpListener::bind(\"127.0.0.1:0\").unwrap();\nlet p = common::port_of(&l);\ndrop(l);",
        // socket2.
        "let s = Socket::new(Domain::IPV4, Type::STREAM, None)?;\ns.bind(&\"127.0.0.1:0\".parse::<std::net::SocketAddr>()?.into())?;\nlet p = s.local_addr()?.as_socket().unwrap().port();\ndrop(s);",
        "s.bind(&SockAddr::from(SocketAddr::from(([127, 0, 0, 1], 0))))?;",
        // SocketAddr constructors and setters.
        "let a = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);",
        "let a = SocketAddrV4::new(loopback, 0);",
        "let a = SocketAddrV6::new(Ipv6Addr::LOCALHOST, 0, 0, 0);",
        "addr.set_port(0);",
        "let a: SocketAddr = ([127, 0, 0, 1], 0).into();",
        // IPv6, names and formatted addresses.
        "TcpListener::bind(\"[::1]:0\").unwrap();",
        "TcpListener::bind(\"localhost:0\").unwrap();",
        "TcpListener::bind(format!(\"{host}:0\")).unwrap();",
        "TcpListener::bind(host.to_string() + \":0\").unwrap();",
        "TcpListener::bind(r#\"127.0.0.1:0\"#).unwrap();",
        // A held listener is still a zero port written outside `ports.rs`.
        "let listener = TcpListener::bind(\"127.0.0.1:0\").await.unwrap();\ntokio::spawn(serve(listener));",
        "CallbackServer::bind(\"127.0.0.1\", 0).unwrap();",
        // A zero formatted in as the port.
        "let a = format!(\"127.0.0.1:{}\", 0);",
        "let a = format!(\"{host}:{}\", 0u16);",
        "let a = format!(\"{host}:{port}\", port = 0);",
        "let a = format!(\"{}:{}\", host, 0);",
        "let port = 0;\nlet a = format!(\"127.0.0.1:{port}\");",
        "TcpListener::bind(format!(\"localhost:{}\", ANY)).unwrap();\nconst ANY: u16 = 0;",
        // A zero in a port field.
        "let cfg = Config { host: \"127.0.0.1\".into(), port: 0 };",
        "let cfg = ListenConfig {\n    listen_port: 0,\n    ..Default::default()\n};",
        // Layout cannot hide it.
        "TcpListener::bind((\n    \"127.0.0.1\",\n    0,\n))",
    ] {
        assert!(!findings(bad).is_empty(), "missed: {bad:?}");
    }
    for fine in [
        "let l = TcpListener::bind(\"127.0.0.1:8080\").unwrap();",
        "let listener = beyond_ai_test_support::ports::listener();",
        "let dead = DeadPort::bind();",
        "// let p = TcpListener::bind(\"127.0.0.1:0\").unwrap();",
        "/* TcpListener::bind(\"127.0.0.1:0\") */",
        // A port-0 address that is not one.
        "let model = \"us.anthropic.claude-haiku-4-5-20251001-v1:0\";",
        "let id = \"functions.read:0\";",
        "assert!(line.contains(r#\"\"input_tokens\":0\"#));",
        "let t = \"12:00:01\";",
        "let ratio = \"1:0\";",
        // A URL naming port 0 is a connect target, not a bind.
        "let url = \"http://127.0.0.1:0/x\";",
        // Zeros that are not ports.
        "assert_eq!(n, 0);",
        "let pair = (name, 0);",
        "call.bind(handle.id.clone());",
        "TcpListener::bind((\"127.0.0.1\", held.port())).unwrap();",
        "let x = vec![(1, 0)];",
        "let c = '\"'; let l = TcpListener::bind(addr).unwrap();",
        // A format string that is not an address, or a port that is not zero.
        "let t = format!(\"{} of {}\", 12, 0);",
        "let mut attempt = 0;\nlet key = format!(\"{seed_base}:{attempt}\");",
        "let a = format!(\"127.0.0.1:{}\", port);",
        "let line = format!(\"line {}:{}\", 0, col);",
        "let a = format!(\"{host}:{port}\", host = 0, port = p);",
        // Fields that are not ports, and port fields that are not zero or are types.
        "let p = Point { x: 0, y: 0 };",
        "let cfg = Config { port: held.port() };",
        "struct Config { port: u16 }",
        "fn f(port: u16) {}",
        // Annotated, on the line or above it.
        "TcpListener::bind(\"127.0.0.1:0\").unwrap(); // port-0: held by the server for its life",
        "// port-0: the child binds it and reports the port it got.\n\"127.0.0.1:0\",",
    ] {
        assert_eq!(findings(fine), Vec::<String>::new(), "{fine:?}");
    }
}
