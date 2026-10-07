//! Test seams stay out of release builds — not just their behaviour, their names too.
//!
//! A seam (`BEYOND_AI_AGENT_TEST_SLOW_CHECKPOINT_MS`, `…_FAIL_CHECKPOINT`, …) changes how the binary
//! behaves when an environment variable is set: fine in a debug test binary, never acceptable in a
//! shipped one. Each lives in a function (or a match arm) under `#[cfg(debug_assertions)]`, with a
//! no-op twin for release, so a release binary has no injection path and carries none of the names.
//! A name written at a call site instead — `delay("BEYOND_AI_AGENT_TEST_…")` into a gated helper —
//! still compiles into release as a string the binary carries.
//!
//! **The authority is the release binary itself**: CI's `build (release)` job runs
//! `scripts/scan-release-seams.sh` on every release binary it built (`mise check:release-seams`),
//! and fails on any `BEYOND_AI_AGENT_TEST_` string the compiler kept, however the source spelled it
//! (`concat!`, a `//` inside a string literal, anything a text heuristic misreads). The source check
//! here is only the fast early warning; the scanner's own behaviour is pinned below.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

/// Where a seam name may appear: in a function, or a match arm, marked `#[cfg(debug_assertions)]`.
/// Returns each occurrence that is not, as `file:line`.
fn ungated_seam_names(path: &str, source: &str) -> Vec<String> {
    let lines: Vec<&str> = source.lines().collect();
    let mut found = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        if !code.contains("BEYOND_AI_AGENT_TEST_") {
            continue;
        }
        // A gated match arm or statement: the attribute sits directly above the start of the arm or
        // statement this line belongs to (walking back over its own continuation lines), not merely
        // somewhere close above — an attribute on the *previous* statement gates that one only.
        let mut start = i;
        while start > 0 {
            let prev = lines[start - 1].split("//").next().unwrap_or("").trim_end();
            let ends_item = prev.is_empty()
                || prev.ends_with(';')
                || prev.ends_with('{')
                || prev.ends_with('}')
                || prev.ends_with(',')
                || prev.trim_start().starts_with("#[");
            if ends_item {
                break;
            }
            start -= 1;
        }
        let arm_gated = start > 0 && lines[start - 1].trim() == "#[cfg(debug_assertions)]";
        // A gated function: the nearest enclosing `fn` (less indented than this line) carries it.
        let indent = line.len() - line.trim_start().len();
        let fn_gated = (0..i)
            .rev()
            .find(|&j| {
                let l = lines[j];
                let t = l.trim_start();
                (l.len() - t.len()) < indent
                    && (t.starts_with("fn ")
                        || t.starts_with("async fn ")
                        || t.starts_with("pub fn ")
                        || t.starts_with("pub(crate) fn ")
                        || t.starts_with("pub async fn ")
                        || t.starts_with("pub(crate) async fn "))
            })
            .is_some_and(|f| {
                lines[..f]
                    .iter()
                    .rev()
                    .take_while(|l| {
                        let t = l.trim();
                        t.starts_with("#[") || t.starts_with("///") || t.starts_with("//")
                    })
                    .any(|l| l.trim() == "#[cfg(debug_assertions)]")
            });
        if !arm_gated && !fn_gated {
            found.push(format!("{path}:{}: {}", i + 1, line.trim()));
        }
    }
    found
}

#[test]
fn every_seam_name_is_compiled_out_of_release_builds() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    let mut dirs = vec![src];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    let mut seams = 0;
    let mut found = Vec::new();
    for path in &files {
        let source = std::fs::read_to_string(path).unwrap();
        seams += source.matches("\"BEYOND_AI_AGENT_TEST_").count();
        found.extend(ungated_seam_names(&path.display().to_string(), &source));
    }
    assert!(seams >= 5, "the scan found the seams ({seams})");
    assert!(
        found.is_empty(),
        "a test seam's name outside `#[cfg(debug_assertions)]` — it ships in release builds:\n{}",
        found.join("\n")
    );
}

/// The check itself: a name at a call site into a gated helper is caught; inside a gated function or
/// arm it is not.
#[test]
fn the_seam_check_catches_a_name_at_a_call_site() {
    let call_site = "\
async fn body() {
    delay(\"BEYOND_AI_AGENT_TEST_SLOW_X_MS\").await;
}
";
    assert_eq!(ungated_seam_names("f.rs", call_site).len(), 1);
    let gated_fn = "\
/// A seam.
#[cfg(debug_assertions)]
async fn slow_x() {
    delay(\"BEYOND_AI_AGENT_TEST_SLOW_X_MS\").await;
}
";
    assert!(ungated_seam_names("f.rs", gated_fn).is_empty());
    let gated_arm = "\
fn dispatch(cmd: &str) {
    match cmd {
        #[cfg(debug_assertions)]
        \"__x\" if std::env::var_os(\"BEYOND_AI_AGENT_TEST_X\").is_some() => {}
        _ => {}
    }
}
";
    assert!(ungated_seam_names("f.rs", gated_arm).is_empty());
    // A gated statement, spanning lines, inside an ungated function.
    let gated_statement = "\
async fn body() {
    #[cfg(debug_assertions)]
    if let Some(ms) = std::env::var(
        \"BEYOND_AI_AGENT_TEST_SLOW_X_MS\",
    ).ok() {}
}
";
    assert!(ungated_seam_names("f.rs", gated_statement).is_empty());
    // The attribute gates the statement under it, not the call site after that.
    let gated_neighbour = "\
async fn body() {
    #[cfg(debug_assertions)]
    let x = 1;
    delay(\"BEYOND_AI_AGENT_TEST_SLOW_X_MS\").await;
}
";
    assert_eq!(ungated_seam_names("f.rs", gated_neighbour).len(), 1);
}

/// The artifact scanner CI trusts: it fails on a binary that carries seam names — this test's own
/// debug build of the agent does, by design — and passes one that does not. A name assembled at
/// compile time (`concat!`) is a contiguous string in the artifact, so it is caught the same way.
#[test]
fn the_release_artifact_scan_fails_on_any_seam_name() {
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/scan-release-seams.sh");
    let scan = |artifact: &std::path::Path| {
        std::process::Command::new("sh")
            .arg(&script)
            .arg(artifact)
            .output()
            .unwrap()
    };
    let debug = scan(std::path::Path::new(env!("CARGO_BIN_EXE_beyond-ai-agent")));
    assert_eq!(debug.status.code(), Some(1), "{debug:?}");
    assert!(
        String::from_utf8_lossy(&debug.stderr).contains("BEYOND_AI_AGENT_TEST_SLOW_CHECKPOINT_MS"),
        "{debug:?}"
    );
    let dir = tempfile::tempdir().unwrap();
    let clean = dir.path().join("clean");
    std::fs::write(
        &clean,
        b"\x7fELF BEYOND_AI_AGENT_MCP_IDLE_SECS nothing to see",
    )
    .unwrap();
    assert_eq!(scan(&clean).status.code(), Some(0));
    let assembled = dir.path().join("assembled");
    std::fs::write(
        &assembled,
        concat!("\x7fELF ", "BEYOND_AI_AGENT_", "TEST_", "SLOW_X_MS").as_bytes(),
    )
    .unwrap();
    assert_eq!(scan(&assembled).status.code(), Some(1));
}
