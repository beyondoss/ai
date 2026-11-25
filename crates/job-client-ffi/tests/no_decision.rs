//! The FFI crate decides nothing. Its job is bytes in, one call into `job_client`, bytes out; a status
//! test, an offset sum, a retry counter or a time constant here would be a second copy of the kernel,
//! outside the proof. This test fails when a source of the crate grows one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;

fn source(name: &str) -> String {
    fs::read_to_string(format!("{}/src/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// The code of a source: comments and string contents removed, `->` and `=>` blanked so that `>` means
/// "greater than" and nothing else.
fn code(src: &str) -> String {
    let mut out = String::new();
    for line in src.lines() {
        let line = line.split("//").next().unwrap_or("");
        let mut in_str = false;
        for c in line.chars() {
            if c == '"' {
                in_str = !in_str;
            } else if !in_str {
                out.push(c);
            }
        }
        out.push('\n');
    }
    out.replace("->", " @ ").replace("=>", " @ ")
}

/// The body of `fn name`, between its braces.
fn body<'a>(code: &'a str, signature: &str) -> &'a str {
    let at = code
        .find(signature)
        .unwrap_or_else(|| panic!("no `{signature}`"));
    let open = at + code[at..].find('{').unwrap();
    let mut depth = 0;
    for (i, c) in code[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &code[open + 1..open + i];
                }
            }
            _ => {}
        }
    }
    panic!("`{signature}` is not closed");
}

/// What this crate may call as a method: the codec's own plumbing. A decision is a method (`max`, `min`,
/// `then`, `cmp`, `saturating_sub`, `clamp`, `contains`, `is_empty`...), so none of them is here.
const METHODS: [&str; 38] = [
    // The reader's and writer's own helpers (`codec.rs`), each a decode of one tagged field.
    "u8",
    "u16",
    "u32",
    "i32",
    "u64",
    "flag",
    "mode",
    "mark",
    "rest",
    "state",
    "event",
    "frame",
    "take",
    "split_first_chunk",
    "split_at_checked",
    "ok_or",
    "get",
    "get_mut",
    "copy_from_slice",
    "extend_from_slice",
    "to_le_bytes",
    "from_le_bytes",
    "push",
    "map",
    "map_err",
    "map_or",
    "map_or_else",
    "unwrap_or",
    "try_from",
    "for_each",
    "iter",
    "len",
    "as_bytes",
    "cast_mut",
    "as_ptr",
    "as_mut_ptr",
    "to_string",
    "checked_sub",
];

/// Macros: `vec!` builds an answer; every other (`matches!`, `assert!`, `debug_assert!`) can decide.
const MACROS: [&str; 1] = ["vec"];

/// Keywords that start a branch or a loop.
const KEYWORDS: [&str; 7] = ["if", "else", "while", "for", "loop", "break", "continue"];

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `code` with every generic argument list removed: `Name<..>` after a type name, `::<..>`, and `fn f<..>`.
fn without_generics(code: &str) -> String {
    let chars: Vec<char> = code.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let generic = chars[i] == '<' && i > 0 && {
            let word: String = chars[..i]
                .iter()
                .rev()
                .take_while(|c| is_ident(**c))
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let after_fn = chars[..i - word.len()]
                .iter()
                .rev()
                .collect::<String>()
                .trim_start()
                .starts_with("nf");
            let upper = word.chars().next().is_some_and(char::is_uppercase);
            upper
                || after_fn
                || (i >= 2 && chars[i - 1] == ':' && chars[i - 2] == ':')
                || word.is_empty() && chars[i - 1] == ':'
        };
        if generic {
            let mut depth = 0;
            while i < chars.len() {
                match chars[i] {
                    '<' => depth += 1,
                    '>' => depth -= 1,
                    _ => {}
                }
                i += 1;
                if depth == 0 {
                    break;
                }
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// What is wrong with `code` (a source with comments and strings removed): empty when it is only the
/// codec's constructs, namely tag matches, the allowed methods, and no operator that computes.
fn lint(code: &str) -> Vec<String> {
    let mut bad = Vec::new();
    let flat = without_generics(&self::code(code));
    let words: Vec<&str> = flat
        .split(|c: char| !is_ident(c))
        .filter(|w| !w.is_empty())
        .collect();
    for k in KEYWORDS {
        if words.contains(&k) {
            bad.push(format!("keyword `{k}`"));
        }
    }
    // Every `.name(` is an allowed method; every `name!` an allowed macro.
    for (at, _) in flat.match_indices('.') {
        let name: String = flat[at + 1..]
            .chars()
            .take_while(|c| is_ident(*c))
            .collect();
        let call = flat[at + 1 + name.len()..].trim_start().starts_with('(');
        if call && !METHODS.contains(&name.as_str()) {
            bad.push(format!("method `.{name}(`"));
        }
    }
    for (at, _) in flat.match_indices('!') {
        let name: String = flat[..at]
            .chars()
            .rev()
            .take_while(|c| is_ident(*c))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let call = !flat[at + 1..].starts_with('=');
        if call && !name.is_empty() && !MACROS.contains(&name.as_str()) {
            bad.push(format!("macro `{name}!`"));
        }
    }
    // Operators. A comparison is `<`, `>`, `==`, `!=`, `<=`, `>=` (generics are gone); arithmetic and
    // bit operators count when binary, i.e. after an operand.
    for op in ["==", "!=", "<", ">", "&&", "||"] {
        if flat.contains(op) {
            bad.push(format!("operator `{op}`"));
        }
    }
    let chars: Vec<char> = flat.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if !"+-*/%&|^".contains(*c) {
            continue;
        }
        let prev = chars[..i].iter().rev().find(|p| !p.is_whitespace());
        // `|x| x` closes its parameters with a `|` that follows an operand: a closure is not a decision.
        let closure_close = *c == '|'
            && chars[..i]
                .iter()
                .rev()
                .take_while(|p| **p != '(' && **p != ',')
                .any(|p| *p == '|');
        if prev.is_some_and(|p| (is_ident(*p) || ")]".contains(*p)) && !closure_close) {
            let near: String = chars[i.saturating_sub(12)..(i + 8).min(chars.len())]
                .iter()
                .collect();
            bad.push(format!("binary `{c}` near `{}`", near.trim()));
        }
    }
    bad
}

#[test]
fn the_sources_hold_only_the_codecs_constructs() {
    for name in ["codec.rs", "abi.rs", "lib.rs"] {
        let bad = lint(&source(name));
        assert!(
            bad.is_empty(),
            "{name}: a decision belongs in job_client: {bad:?}"
        );
    }
}

#[test]
fn the_lint_is_red_on_each_way_a_decision_gets_in() {
    // Each of these is a decision written the way the old lexical rules missed.
    let cases = [
        ("a>b", "operator `>`"),
        ("let m = a.max(b);", "method `.max(`"),
        ("let m = a.min(b);", "method `.min(`"),
        ("let m = a.saturating_sub(b);", "method `.saturating_sub(`"),
        ("let m = a.clamp(1, 2);", "method `.clamp(`"),
        (
            "let ok = matches!(s.phase, Phase::Done);",
            "macro `matches!`",
        ),
        ("assert!(x);", "macro `assert!`"),
        ("if x { y }", "keyword `if`"),
        ("while x { y }", "keyword `while`"),
        ("let t = a+b;", "binary `+`"),
        ("let t = a-b;", "binary `-`"),
        ("let t = (a)*(b);", "binary `*`"),
        ("let t = a&b;", "binary `&`"),
        ("let t = a == b;", "operator `==`"),
        ("let t = a!=b;", "operator `!=`"),
        ("let t = a && b;", "operator `&&`"),
        ("let t = a.then(|| 1);", "method `.then(`"),
        ("let t = x.cmp(&y);", "method `.cmp(`"),
        ("match n { 1..=9 if n > 5 => 1, _ => 2 }", "keyword `if`"),
    ];
    for (src, want) in cases {
        let bad = lint(src);
        assert!(
            bad.iter().any(|b| b.starts_with(want)),
            "`{src}` should be red with `{want}`, got {bad:?}"
        );
    }
    // And the codec's own shapes stay green.
    for ok in [
        "fn take<const N: usize>(&mut self) -> R<[u8; N]> { let (h, t) = self.0.split_first_chunk::<N>().ok_or(BAD)?; Ok(*h) }",
        "let v = vec![0]; v.push(1); out.get_mut(..n).ok_or(E)?.copy_from_slice(&a);",
        "match self.u8()? { 0 => Ok(A), _ => Err(BAD_INPUT) }",
        "let s = State { phase, flow, ..x }; let u = u8::from(s.flag) as usize;",
        "call(a, b).map_or_else(|code| code, |len| i32::try_from(len).unwrap_or(E))",
    ] {
        assert_eq!(lint(ok), Vec::<String>::new(), "`{ok}` is codec-shaped");
    }
}

#[test]
fn the_sources_name_no_status_and_no_constant_of_the_kernel() {
    for name in ["codec.rs", "abi.rs"] {
        let src = code(&source(name));
        // Any three-digit number (an HTTP status, a time) and the kernel's own constants.
        let bytes = src.as_bytes();
        for (i, w) in bytes.windows(3).enumerate() {
            let digits = w.iter().all(u8::is_ascii_digit);
            let before = i.checked_sub(1).map(|j| bytes[j]);
            let after = bytes.get(i + 3).copied();
            let lone = !before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
                && !after.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_');
            assert!(
                !(digits && lone),
                "{name} holds a number of three digits near byte {i}"
            );
        }
        for k in ["RECONNECTING_AFTER", "BACKOFF_CAP_MS", "SILENCE_FLOOR_MS"] {
            assert!(!src.contains(k), "{name} uses `{k}`");
        }
    }
}

#[test]
fn every_export_is_one_call_into_the_codec() {
    let abi = code(&source("abi.rs"));
    let mut seen = 0;
    for (at, _) in abi.match_indices("pub extern  fn ") {
        let rest = &abi[at + "pub extern  fn ".len()..];
        let name = rest.split('(').next().unwrap().to_string();
        let b = body(&abi, &format!("pub extern  fn {name}("))
            .trim()
            .to_string();
        match name.as_str() {
            // Constant, and the two memory functions a WASM host needs.
            "jc_abi_version" => assert_eq!(b, "ABI_VERSION"),
            "jc_alloc" | "jc_free" => {}
            _ => {
                let want = format!(
                    "io(inp, n, out, cap, codec::{})",
                    name.trim_start_matches("jc_")
                );
                let alt = b
                    .strip_prefix("io(inp, n, out, cap, codec::")
                    .and_then(|r| r.strip_suffix(')'));
                assert!(
                    alt.is_some()
                        && !alt
                            .unwrap()
                            .contains(|c: char| !(c.is_ascii_alphanumeric() || c == '_')),
                    "{name} is `{b}`, not a single `{want}`-shaped call"
                );
            }
        }
        seen += 1;
    }
    assert_eq!(
        seen, 10,
        "an export was added or removed: update this test with it"
    );
}

#[test]
fn only_the_kernel_is_called_and_it_is_the_only_dependency() {
    let src = code(&source("codec.rs"));
    // Every path into the kernel is one of these, each a proven function or a constructor of its types.
    let allowed = [
        "job_client::decode_bytes",
        "job_client::encode_bytes",
        "job_client::Cursor::new",
        "job_client::step::",
    ];
    let mut rest = src.as_str();
    while let Some(at) = rest.find("job_client::") {
        let path: String = rest[at..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
            .collect();
        assert!(
            allowed.contains(&path.as_str()),
            "codec.rs reaches into the kernel at `{path}`"
        );
        rest = &rest[at + path.len()..];
    }
    // The one `use` of the kernel's step module: its types, and the three proven functions.
    let uses = src
        .split("use job_client::step::{")
        .nth(1)
        .unwrap()
        .split('}')
        .next()
        .unwrap()
        .to_string();
    let names: Vec<&str> = uses
        .split(',')
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .collect();
    let known = [
        "Action",
        "Class",
        "Event",
        "Flow",
        "Frame",
        "Mark",
        "Mode",
        "Op",
        "Phase",
        "State",
        "classify as k_classify",
        "silence_ms",
        "step as k_step",
    ];
    assert!(
        names.iter().all(|n| known.contains(n)),
        "codec.rs imports from the kernel: {names:?}"
    );
    let manifest =
        fs::read_to_string(format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let deps = manifest.split("[dependencies]").nth(1).unwrap();
    assert_eq!(
        deps.trim(),
        "beyond-job-client = { path = \"../job-client\" }"
    );
}
