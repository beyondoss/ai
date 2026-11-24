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
    out.replace("->", "  ").replace("=>", "  ")
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

const FORBIDDEN_WORDS: [&str; 6] = ["if ", "while ", "for ", "loop ", "loop{", ".cmp("];
const FORBIDDEN_OPS: [&str; 10] = [
    "==", "!=", "<=", ">=", " < ", " > ", " + ", " - ", " * ", " / ",
];

#[test]
fn the_sources_hold_no_branch_comparison_or_arithmetic() {
    for name in ["codec.rs", "abi.rs", "lib.rs"] {
        let src = code(&source(name));
        // `jc_free`/`jc_alloc` and `io` are memory plumbing, held to the same rule: they have no branch either.
        for w in FORBIDDEN_WORDS {
            assert!(
                !src.contains(w),
                "{name} has `{}`: a decision belongs in job_client",
                w.trim()
            );
        }
        for op in FORBIDDEN_OPS {
            assert!(
                !src.contains(op),
                "{name} has `{}`: a decision belongs in job_client",
                op.trim()
            );
        }
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
        for k in [
            "BUDGET",
            "BACKOFF_CAP_MS",
            "SILENCE_FLOOR_MS",
            "misses",
            "until_ms +",
        ] {
            let hits = src.matches(k).count();
            // `misses` and `until_ms` are fields the state record copies byte for byte, never read for a decision.
            let allowed = if k == "misses" { 2 } else { 0 };
            assert!(hits <= allowed, "{name} uses `{k}`");
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
