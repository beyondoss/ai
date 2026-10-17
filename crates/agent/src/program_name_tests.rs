//! No user-facing string names the program as anything but [`crate::PROGRAM`].
//!
//! A hint like "run `agent login`" sends a user to a command that doesn't exist: the binary is
//! `beyond-ai-agent`. Every hint formats `PROGRAM` in instead. This scans the crate's sources for a
//! string that hard-codes a command line under another name.

use std::path::Path;

/// The subcommands a hint might tell a user to run.
const SUBCOMMANDS: &[&str] = &[
    "auth-status",
    "clear-trust",
    "export",
    "list-models",
    "login",
    "logout",
    "mcp-login",
    "mcp-logout",
    "run",
    "serve",
    "settings",
    "tools",
    "trust",
    "trust-status",
    "untrust",
];

/// Names a hint could plausibly use instead of the real one.
const WRONG_NAMES: &[&str] = &["agent", "ai-agent", "beyond-agent", "beyond"];

fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Every `` `<wrong-name> <subcommand>`` inside a string literal on a non-comment line.
fn offenders(source: &str) -> Vec<String> {
    let mut found = Vec::new();
    for (n, line) in source.lines().enumerate() {
        let code = line.trim_start();
        if code.starts_with("//") {
            continue;
        }
        // Only text inside a string literal is user-facing.
        let Some(open) = code.find('"') else {
            continue;
        };
        let text = &code[open..];
        for name in WRONG_NAMES {
            for sub in SUBCOMMANDS {
                let bad = format!("`{name} {sub}");
                if text.match_indices(&bad).any(|(i, _)| {
                    text[i + bad.len()..]
                        .chars()
                        .next()
                        .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '-'))
                }) {
                    found.push(format!("line {}: {}", n + 1, line.trim()));
                }
            }
        }
    }
    found
}

#[test]
fn the_program_name_is_the_binary_name() {
    assert_eq!(crate::PROGRAM, "beyond-ai-agent");
}

#[test]
fn no_string_names_the_program_any_other_way() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    let this = Path::new(file!()).file_name().unwrap();
    let mut bad = Vec::new();
    for file in files {
        if file.file_name() == Some(this) {
            continue;
        }
        let source = std::fs::read_to_string(&file).unwrap();
        for hit in offenders(&source) {
            bad.push(format!("{}: {hit}", file.display()));
        }
    }
    assert!(
        bad.is_empty(),
        "user-facing strings name a command that doesn't exist; format `crate::PROGRAM` in \
         instead:\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_scan_catches_a_hard_coded_name_and_ignores_comments() {
    assert_eq!(offenders("        \"run `agent login` first\",\n").len(), 1);
    assert!(offenders("    // run `agent login` first\n").is_empty());
    assert!(offenders("        \"run `{} login` first\",\n").is_empty());
    // `agent runner` is not `agent run`.
    assert!(offenders("        \"the `agent runner` crate\",\n").is_empty());
}
