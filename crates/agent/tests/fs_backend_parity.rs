//! Differential parity: `LocalFs` vs `ShellFs`, over the *same* files.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//!
//!
//! This is the test that justifies the [`FsBackend`] seam existing. A trait with one implementation
//! proves nothing — it can always be shaped around its only caller. So `ShellFs` runs here over
//! [`RealRunner`], against a real fixture tree on the host, and every assertion compares the full
//! `ToolOutput` text the *model* would see against what the in-process ripgrep walker produces.
//! No VM, no container, no infrastructure: if the seam is wrong, this file fails.
//!
//! ## Two rungs, deliberately
//!
//! `ShellFs` is exercised twice — once with `rg` (whatever the host actually has) and once with
//! `Capabilities { rg: false }`, which forces the POSIX `grep` fallback even on a box that has
//! ripgrep installed. Without that second rung the fallback would only ever be tested by accident, on
//! machines nobody develops on. Where the two rungs genuinely cannot agree — `.gitignore` — the
//! difference is asserted as a *number*, not skipped, so its cost is something a reviewer approved.
//!
//! ## What is deliberately not compared byte-for-byte
//!
//! A **truncated** `LocalFs` result is documented as non-deterministic: the parallel walk quits the
//! moment it has `limit` matches, so which matches survive depends on thread scheduling. `ShellFs`
//! runs the search to completion and truncates afterwards, so it is strictly *more* deterministic.
//! Comparing their contents on a truncated query would be asserting that a documented
//! non-determinism happens to land a particular way. Those cases assert the match *count* and the
//! truncation notice instead.

use std::path::Path;
use std::sync::Arc;

use agent_core::Tool;
use beyond_ai_agent::tools::exec::RealRunner;
use beyond_ai_agent::tools::fs::FsBackend;
use beyond_ai_agent::tools::fs::local::LocalFs;
use beyond_ai_agent::tools::fs::shell::{Capabilities, SearchEngine, ShellFs};
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::{Value, json};
use tempfile::TempDir;

/// A registry built against `backend`, so a test drives the *real* tool the registry hands the model
/// rather than a hand-constructed one — the same assembly path production uses.
fn registry_for(backend: Option<Arc<dyn FsBackend>>) -> agent_core::ToolRegistry {
    default_registry_with_config(&ToolConfig {
        fs_backend: backend,
        ..ToolConfig::new()
    })
}

fn tool_of(reg: &agent_core::ToolRegistry, name: &str) -> Arc<dyn Tool> {
    reg.get(name).expect("tool must be registered")
}

/// A tree of the cases that break naive implementations. Built fresh per test so a test that creates
/// a FIFO or a broken symlink can't leak state into another.
///
/// Deliberately **not** inside a git repository: that is the configuration where `LocalFs` applies
/// `require_git(false)` and `ShellFs` passes `--no-require-git`, so it is the one that actually
/// exercises the "honor .gitignore even outside a repo" divergence from stock ripgrep.
fn fixture() -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path();

    // Ordinary source-shaped files, several so path sorting is observable.
    std::fs::write(p.join("alpha.rs"), "fn one() { NEEDLE }\nfn two() {}\n").unwrap();
    std::fs::write(p.join("beta.rs"), "// nothing here\n").unwrap();
    std::fs::write(p.join("gamma.txt"), "NEEDLE in text\nand NEEDLE again\n").unwrap();

    // Context: a match with distinct lines either side, for -B/-A parity.
    std::fs::write(
        p.join("ctx.txt"),
        "before-line\nNEEDLE middle\nafter-line\nfiller\nfiller\nsecond NEEDLE\ntail-line\n",
    )
    .unwrap();

    // .gitignore'd subtree containing a real match — the headline divergence.
    std::fs::write(p.join(".gitignore"), "ignored/\n").unwrap();
    std::fs::create_dir(p.join("ignored")).unwrap();
    std::fs::write(p.join("ignored/hidden.rs"), "NEEDLE inside ignored\n").unwrap();

    // Filenames that break output parsing.
    std::fs::create_dir(p.join("sub")).unwrap();
    std::fs::write(p.join("sub/has space.txt"), "NEEDLE with space\n").unwrap();
    std::fs::write(p.join("sub/we\nird.txt"), "NEEDLE with newline\n").unwrap();
    std::fs::write(p.join("sub/odd:colon.txt"), "NEEDLE with colon\n").unwrap();
    // A path made of shell metacharacters. If anything ever builds a command *string*, this is the
    // fixture that turns that mistake into a deleted machine instead of a silent bug.
    // No `/` in the name — that is a path separator, not a shell character. Everything else here is:
    // a quote, a command separator, a substitution, a glob, and a comment.
    std::fs::write(p.join("sub/'; rm -rf . $(id) * #.txt"), "NEEDLE quoted\n").unwrap();

    // Encoding edge cases.
    std::fs::write(p.join("crlf.txt"), "NEEDLE crlf line\r\nsecond\r\n").unwrap();
    std::fs::write(p.join("latin1.txt"), b"caf\xe9 NEEDLE here\n").unwrap();
    std::fs::write(p.join("empty.txt"), "").unwrap();
    // A NUL marks the file binary; both engines must skip it rather than emit garbage.
    std::fs::write(p.join("blob.bin"), b"\x00\x00 NEEDLE \x00binary").unwrap();

    // A dotfile — `hidden(false)` / `--hidden` means these are searched, unlike stock ripgrep.
    std::fs::write(p.join(".dotfile"), "NEEDLE in a dotfile\n").unwrap();

    dir
}

/// Add a symlink and a broken symlink. Separate from [`fixture`] because a broken symlink makes the
/// walk report an error, which is its own assertion rather than background noise in every test.
fn add_symlinks(p: &Path) {
    std::os::unix::fs::symlink(p.join("alpha.rs"), p.join("link-to-alpha.rs")).unwrap();
    std::os::unix::fs::symlink(p.join("does-not-exist"), p.join("broken-link.rs")).unwrap();
}

fn local_backend() -> Arc<dyn FsBackend> {
    Arc::new(LocalFs::new())
}

/// `ShellFs` with whatever the host has — the good rung, expected to match `LocalFs` exactly.
async fn shell_best_backend() -> Arc<dyn FsBackend> {
    let backend = ShellFs::connect(Arc::new(RealRunner)).await;
    assert_eq!(
        backend.capabilities().search_engine(),
        SearchEngine::Ripgrep,
        "this host has no `rg`, so the good rung is untested here — install ripgrep to get real \
         parity coverage rather than a silently degraded run"
    );
    Arc::new(backend)
}

/// `ShellFs` with `rg` forced absent — the fallback rung, exercised even on a box that has ripgrep.
fn shell_posix_backend() -> Arc<dyn FsBackend> {
    Arc::new(ShellFs::with_capabilities(
        Arc::new(RealRunner),
        Capabilities { rg: false },
    ))
}

async fn run_tool(backend: Option<Arc<dyn FsBackend>>, name: &str, input: Value) -> String {
    let reg = registry_for(backend);
    tool_of(&reg, name)
        .run(input)
        .await
        .unwrap_or_else(|e| panic!("{name} failed: {e}"))
        .text
}

async fn run_tool_err(backend: Option<Arc<dyn FsBackend>>, name: &str, input: Value) -> String {
    let reg = registry_for(backend);
    match tool_of(&reg, name).run(input).await {
        Ok(o) => panic!("{name} unexpectedly succeeded: {}", o.text),
        Err(e) => e.to_string(),
    }
}

/// Assert both backends produce byte-identical model-visible output for `tool`.
async fn assert_tool_parity(tool: &str, mut input: Value, dir: &TempDir) -> String {
    if input.get("path").is_none() {
        input["path"] = json!(dir.path().to_str().unwrap());
    }
    let l = run_tool(Some(local_backend()), tool, input.clone()).await;
    let s = run_tool(Some(shell_best_backend().await), tool, input.clone()).await;
    assert_eq!(
        l, s,
        "{tool}: LocalFs and ShellFs disagreed.\n--- local ---\n{l}\n--- shell ---\n{s}"
    );
    l
}

/// The `grep`-specific shorthand the original suite was written against.
async fn assert_parity(dir: &TempDir, input: Value) -> String {
    assert_tool_parity("grep", input, dir).await
}

fn match_lines(out: &str) -> usize {
    out.lines()
        .filter(|l| l.contains(": ") && !l.starts_with('['))
        .count()
}

// ---------------------------------------------------------------- parity

#[tokio::test]
async fn plain_search_is_identical_across_backends() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "NEEDLE" })).await;
    assert!(out.contains("alpha.rs"), "{out}");
    assert!(out.contains("gamma.txt"), "{out}");
}

#[tokio::test]
async fn filenames_with_spaces_newlines_colons_and_shell_metacharacters_are_identical() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "NEEDLE" })).await;
    assert!(out.contains("has space.txt"), "space: {out}");
    assert!(out.contains("odd:colon.txt"), "colon: {out}");
    assert!(out.contains("we\nird.txt"), "newline: {out}");
    assert!(out.contains("rm -rf"), "metacharacters: {out}");
    // The fixture must still be intact — proof the metacharacter path was an argument, not a command.
    assert!(dir.path().join("alpha.rs").exists(), "the tree was damaged");
}

#[tokio::test]
async fn encoding_edge_cases_are_identical() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "NEEDLE" })).await;
    assert!(out.contains("crlf.txt"), "crlf: {out}");
    assert!(out.contains("latin1.txt"), "non-utf8: {out}");
    assert!(
        !out.contains("blob.bin"),
        "the binary file must be skipped by both: {out}"
    );
}

#[tokio::test]
async fn gitignored_matches_are_excluded_by_both_backends_outside_a_repo() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "NEEDLE" })).await;
    assert!(
        !out.contains("hidden.rs"),
        "a .gitignore'd match must not appear, even outside a git repo: {out}"
    );
}

#[tokio::test]
async fn dotfiles_are_searched_by_both_backends() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "NEEDLE" })).await;
    assert!(out.contains(".dotfile"), "{out}");
}

#[tokio::test]
async fn context_lines_are_identical() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "NEEDLE middle", "context": 1 })).await;
    assert!(out.contains("before-line"), "{out}");
    assert!(out.contains("after-line"), "{out}");
    // Context lines use the `-` separator, matches use `:` — the distinction must survive the wire.
    assert!(out.contains("-2- ") || out.contains("-1- "), "{out}");
}

#[tokio::test]
async fn asymmetric_before_and_after_context_is_identical() {
    let dir = fixture();
    assert_parity(
        &dir,
        json!({ "pattern": "second NEEDLE", "before": 2, "after": 0 }),
    )
    .await;
}

#[tokio::test]
async fn glob_filtering_is_identical() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "NEEDLE", "glob": "*.rs" })).await;
    assert!(out.contains("alpha.rs"), "{out}");
    assert!(!out.contains("gamma.txt"), "{out}");
}

#[tokio::test]
async fn negated_glob_filtering_is_identical() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "NEEDLE", "glob": "!*.rs" })).await;
    assert!(!out.contains("alpha.rs"), "{out}");
    assert!(out.contains("gamma.txt"), "{out}");
}

#[tokio::test]
async fn literal_mode_is_identical() {
    let dir = fixture();
    std::fs::write(dir.path().join("meta.txt"), "a.b(c) NEEDLE\naXb(c)\n").unwrap();
    assert_parity(&dir, json!({ "pattern": "a.b(c)", "literal": true })).await;
}

#[tokio::test]
async fn ignore_case_is_identical() {
    let dir = fixture();
    std::fs::write(dir.path().join("case.txt"), "needle lower\nNeEdLe mixed\n").unwrap();
    assert_parity(&dir, json!({ "pattern": "needle", "ignore_case": true })).await;
}

#[tokio::test]
async fn no_matches_is_reported_identically() {
    let dir = fixture();
    let out = assert_parity(&dir, json!({ "pattern": "zzz-no-such-thing" })).await;
    assert!(out.starts_with("no matches for"), "{out}");
}

#[tokio::test]
async fn symlinks_do_not_diverge() {
    let dir = fixture();
    add_symlinks(dir.path());
    // Neither backend follows symlinks into files by default, and a broken one must not crash either.
    assert_parity(&dir, json!({ "pattern": "NEEDLE" })).await;
}

#[tokio::test]
async fn a_fifo_does_not_hang_or_diverge() {
    let dir = fixture();
    let fifo = dir.path().join("pipe.fifo");
    let ok = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipping: mkfifo unavailable");
        return;
    }
    // The real risk is a backend blocking forever on open(2). The test timing out *is* the assertion.
    assert_parity(&dir, json!({ "pattern": "NEEDLE" })).await;
}

// ------------------------------------------------------ read / ls / find

#[tokio::test]
async fn read_returns_identical_text_across_backends() {
    let dir = fixture();
    let p = dir.path().join("ctx.txt");
    assert_tool_parity("read", json!({ "path": p.to_str().unwrap() }), &dir).await;
}

#[tokio::test]
async fn read_windows_identically_with_offset_and_limit() {
    let dir = fixture();
    let p = dir.path().join("ctx.txt");
    let out = assert_tool_parity(
        "read",
        json!({ "path": p.to_str().unwrap(), "offset": 2, "limit": 3 }),
        &dir,
    )
    .await;
    assert!(out.contains("NEEDLE middle"), "{out}");
    assert!(
        !out.contains("tail-line"),
        "the window must not overrun: {out}"
    );
}

#[tokio::test]
async fn read_reports_an_empty_file_identically() {
    let dir = fixture();
    let p = dir.path().join("empty.txt");
    let out = assert_tool_parity("read", json!({ "path": p.to_str().unwrap() }), &dir).await;
    assert_eq!(out, "(empty file)");
}

#[tokio::test]
async fn read_handles_crlf_and_non_utf8_identically() {
    let dir = fixture();
    for name in ["crlf.txt", "latin1.txt"] {
        let p = dir.path().join(name);
        assert_tool_parity("read", json!({ "path": p.to_str().unwrap() }), &dir).await;
    }
}

#[tokio::test]
async fn read_of_a_binary_file_is_identical() {
    // Not an image and not valid UTF-8 — the lossy-decode path, which must agree byte for byte.
    let dir = fixture();
    let p = dir.path().join("blob.bin");
    assert_tool_parity("read", json!({ "path": p.to_str().unwrap() }), &dir).await;
}

#[tokio::test]
async fn read_of_a_png_returns_an_identical_image_attachment() {
    // The one path where bytes, not text, cross the backend. `ExecResult.stdout` is a `String`, so a
    // naive shell read would lossily mangle every non-UTF-8 byte — this is the test that proves the
    // base64 round trip actually preserves the image.
    let dir = fixture();
    let png = dir.path().join("pic.png");
    let img = image::RgbaImage::from_pixel(8, 8, image::Rgba([10, 200, 30, 255]));
    img.save_with_format(&png, image::ImageFormat::Png).unwrap();

    let input = json!({ "path": png.to_str().unwrap() });
    let reg_l = registry_for(Some(local_backend()));
    let reg_s = registry_for(Some(shell_best_backend().await));
    let l = tool_of(&reg_l, "read").run(input.clone()).await.unwrap();
    let s = tool_of(&reg_s, "read").run(input.clone()).await.unwrap();

    assert_eq!(l.images.len(), 1, "the local read must attach the image");
    assert_eq!(s.images.len(), 1, "the shell read must attach the image");
    assert_eq!(l.text, s.text);
    assert_eq!(
        l.images[0], s.images[0],
        "the image bytes must survive the base64 round trip exactly"
    );
}

#[tokio::test]
async fn read_of_a_missing_file_errors_identically() {
    let dir = fixture();
    let p = dir.path().join("nope.txt");
    let input = json!({ "path": p.to_str().unwrap() });
    let l = run_tool_err(Some(local_backend()), "read", input.clone()).await;
    let s = run_tool_err(Some(shell_best_backend().await), "read", input).await;
    for e in [&l, &s] {
        assert!(e.contains("nope.txt"), "the error must name the path: {e}");
    }
}

#[tokio::test]
async fn ls_lists_identically_including_directory_suffixes() {
    let dir = fixture();
    let out = assert_tool_parity("ls", json!({}), &dir).await;
    assert!(
        out.contains("sub/"),
        "a directory must carry its suffix: {out}"
    );
    assert!(out.contains("alpha.rs"), "{out}");
    assert!(
        !out.contains(".gitignore"),
        "dotfiles are hidden by default: {out}"
    );
}

#[tokio::test]
async fn ls_shows_dotfiles_identically_with_all() {
    let dir = fixture();
    let out = assert_tool_parity("ls", json!({ "all": true }), &dir).await;
    assert!(out.contains(".gitignore"), "{out}");
    assert!(out.contains(".dotfile"), "{out}");
}

#[tokio::test]
async fn ls_errors_identically_on_a_file_and_on_a_missing_path() {
    let dir = fixture();
    for (path, needle) in [
        (dir.path().join("alpha.rs"), "Not a directory"),
        (dir.path().join("nope"), "Path not found"),
    ] {
        let input = json!({ "path": path.to_str().unwrap() });
        let l = run_tool_err(Some(local_backend()), "ls", input.clone()).await;
        let s = run_tool_err(Some(shell_best_backend().await), "ls", input).await;
        assert_eq!(l, s, "ls error text must match for {}", path.display());
        assert!(l.contains(needle), "expected {needle:?}, got: {l}");
    }
}

#[tokio::test]
async fn ls_sorts_awkward_names_identically() {
    let dir = fixture();
    let out = assert_tool_parity(
        "ls",
        json!({ "path": dir.path().join("sub").to_str().unwrap() }),
        &dir,
    )
    .await;
    assert!(out.contains("has space.txt"), "{out}");
    assert!(out.contains("odd:colon.txt"), "{out}");
}

#[tokio::test]
async fn find_matches_identically_by_basename_glob() {
    let dir = fixture();
    let out = assert_tool_parity("find", json!({ "pattern": "*.rs" }), &dir).await;
    assert!(out.contains("alpha.rs"), "{out}");
    assert!(out.contains("beta.rs"), "{out}");
    assert!(!out.contains("hidden.rs"), "gitignored: {out}");
}

#[tokio::test]
async fn find_matches_directories_identically() {
    // The divergence I closed by deriving directories from the file listing's ancestors — without it
    // `find "sub"` would return nothing on the shell backend while returning a match locally.
    let dir = fixture();
    let out = assert_tool_parity("find", json!({ "pattern": "sub" }), &dir).await;
    assert!(
        out.contains("sub/"),
        "a directory match must appear with its trailing slash: {out}"
    );
}

#[tokio::test]
async fn find_matches_identically_by_path_glob() {
    let dir = fixture();
    assert_tool_parity("find", json!({ "pattern": "sub/*.txt" }), &dir).await;
}

#[tokio::test]
async fn find_reports_no_match_identically() {
    let dir = fixture();
    let out = assert_tool_parity("find", json!({ "pattern": "*.nosuchext" }), &dir).await;
    assert!(out.starts_with("no files matching"), "{out}");
}

// ------------------------------------------------------- write / edit

#[tokio::test]
async fn write_then_read_round_trips_identically_across_backends() {
    // Written by one backend, read back by the same one — then the two results compared. This is the
    // test that would catch a `write` that silently mangled content on its way through base64.
    let body = "line one\nline two — with an em dash and a ünïcödé tail\n";
    let mut outs = Vec::new();
    for backend in [local_backend(), shell_best_backend().await] {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("deep").join("nested").join("out.txt");
        let reg = registry_for(Some(backend));
        let wrote = tool_of(&reg, "write")
            .run(json!({ "path": p.to_str().unwrap(), "content": body }))
            .await
            .unwrap()
            .text;
        assert!(wrote.contains("wrote"), "{wrote}");
        // Parent directories must have been created by the backend, not assumed to exist.
        assert_eq!(std::fs::read_to_string(&p).unwrap(), body);
        outs.push(
            tool_of(&reg, "read")
                .run(json!({ "path": p.to_str().unwrap() }))
                .await
                .unwrap()
                .text,
        );
    }
    assert_eq!(
        outs[0], outs[1],
        "read-back text must match across backends"
    );
}

#[tokio::test]
async fn write_preserves_bytes_that_are_not_valid_utf8_neighbours() {
    // Content with characters that survive base64 but would be destroyed by a naive text pipeline.
    let body = "tab\there\nnull-adjacent: \u{fffd}\nquote: \" backslash: \\ dollar: $HOME\n";
    for backend in [local_backend(), shell_best_backend().await] {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tricky.txt");
        let reg = registry_for(Some(backend));
        tool_of(&reg, "write")
            .run(json!({ "path": p.to_str().unwrap(), "content": body }))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            body,
            "content must survive the backend verbatim"
        );
    }
}

#[tokio::test]
async fn write_refuses_a_read_only_file_identically() {
    use std::os::unix::fs::PermissionsExt;
    for backend in [local_backend(), shell_best_backend().await] {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ro.txt");
        std::fs::write(&p, "original").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        let err = run_tool_err(
            Some(backend),
            "write",
            json!({ "path": p.to_str().unwrap(), "content": "new" }),
        )
        .await;
        assert!(err.contains("not writable"), "got: {err}");
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "original",
            "a refused write must not touch the file"
        );
    }
}

#[tokio::test]
async fn edit_applies_identically_across_backends() {
    for backend in [local_backend(), shell_best_backend().await] {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("src.rs");
        std::fs::write(&p, "fn main() {\n    let quick = 1;\n}\n").unwrap();
        let reg = registry_for(Some(backend));
        let out = tool_of(&reg, "edit")
            .run(json!({
                "path": p.to_str().unwrap(),
                "old_string": "quick",
                "new_string": "slow",
            }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("1 replacement"), "{out}");
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "fn main() {\n    let slow = 1;\n}\n"
        );
    }
}

#[tokio::test]
async fn edit_preserves_crlf_line_endings_across_backends() {
    for backend in [local_backend(), shell_best_backend().await] {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("crlf.txt");
        std::fs::write(&p, "alpha\r\nbeta\r\ngamma\r\n").unwrap();
        let reg = registry_for(Some(backend));
        tool_of(&reg, "edit")
            .run(json!({
                "path": p.to_str().unwrap(),
                "old_string": "beta",
                "new_string": "delta",
            }))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "alpha\r\ndelta\r\ngamma\r\n",
            "the file's original line endings must survive the edit"
        );
    }
}

#[tokio::test]
async fn edit_reports_a_missing_match_identically() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("src.rs");
    std::fs::write(&p, "fn main() {}\n").unwrap();
    let input = json!({
        "path": p.to_str().unwrap(),
        "old_string": "not present anywhere",
        "new_string": "x",
    });
    let l = run_tool_err(Some(local_backend()), "edit", input.clone()).await;
    let s = run_tool_err(Some(shell_best_backend().await), "edit", input).await;
    assert_eq!(l, s, "edit's not-found error must be identical");
}

// ------------------------------------------------------ the model's view

#[tokio::test]
async fn the_model_sees_a_byte_identical_toolset_whichever_backend_is_configured() {
    // The whole contract in one assertion. `ToolRegistry::definitions()` is what becomes the tool
    // block in the request, and the Anthropic dialect anchors a prompt-cache breakpoint on it — so if
    // attaching a backend changed a single byte here, every turn would cold-miss the cache *and* the
    // model would be looking at a different toolset than the one it was tuned against.
    let local = registry_for(Some(local_backend())).definitions();
    let shell = registry_for(Some(shell_best_backend().await)).definitions();
    let none = registry_for(None).definitions();

    let json = |defs: &[agent_core::ToolDef]| serde_json::to_string(defs).unwrap();
    assert_eq!(
        json(&local),
        json(&shell),
        "the advertised toolset must not depend on which backend is configured"
    );
    assert_eq!(
        json(&none),
        json(&local),
        "an unconfigured backend must advertise exactly what the explicit local one does"
    );
    // And the names really are the ones the tools are known by, not an accidentally empty set.
    let names: Vec<&str> = local.iter().map(|d| d.name.as_str()).collect();
    for expected in ["read", "write", "edit", "ls", "grep", "find"] {
        assert!(names.contains(&expected), "missing {expected}: {names:?}");
    }
}

// ------------------------------------------------- documented divergence

#[tokio::test]
async fn the_posix_fallback_diverges_in_exactly_two_named_ways() {
    // The measured cost of "use whatever the box has". Both divergences are asserted by name so that
    // if either changes — a better fallback, or a *new* third divergence — this test fails and
    // somebody has to look, rather than the difference quietly widening.
    let dir = fixture();
    let input = json!({ "pattern": "NEEDLE", "path": dir.path().to_str().unwrap(), "limit": 500 });

    let good = run_tool(Some(shell_best_backend().await), "grep", input.clone()).await;
    let fallback = run_tool(Some(shell_posix_backend()), "grep", input.clone()).await;

    // 1. FALSE POSITIVES: no `.gitignore` awareness. On a real repo this is the difference between a
    //    clean result and one buried under `target/` and `node_modules/` — the loud, obvious cost.
    assert!(
        !good.contains("hidden.rs"),
        "rg must honor .gitignore: {good}"
    );
    assert!(
        fallback.contains("hidden.rs"),
        "POSIX grep is expected to walk the ignored tree: {fallback}"
    );

    // 2. FALSE NEGATIVES: `grep -I` classifies any file containing non-UTF-8 bytes as "binary" and
    //    skips it, whereas ripgrep (and `LocalFs`, via `BinaryDetection::quit(b'\0')`) only skips on
    //    an actual NUL. A latin-1 source file therefore *silently loses its matches* on the fallback.
    //    This is the worse of the two: an extra irrelevant hit is visible, a missing hit is not.
    assert!(
        good.contains("latin1.txt"),
        "rg must match inside a non-UTF-8 file: {good}"
    );
    assert!(
        !fallback.contains("latin1.txt"),
        "POSIX grep -I is expected to drop the non-UTF-8 file — if this now passes, the fallback \
         got better and this test should be updated deliberately: {fallback}"
    );

    // 3. Nothing else. Strip the two known-divergent files from both sides and the remainder must be
    //    byte-identical, which is what makes the two assertions above a complete inventory rather
    //    than the two divergences somebody happened to notice.
    let strip = |s: &str| {
        s.lines()
            .filter(|l| !l.contains("hidden.rs") && !l.contains("latin1.txt"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(
        strip(&good),
        strip(&fallback),
        "outside the two named divergences the rungs must agree exactly\n--- rg ---\n{good}\n\
         --- posix ---\n{fallback}"
    );
}

#[tokio::test]
async fn the_posix_fallback_still_agrees_on_everything_gitignore_does_not_touch() {
    // Scoped to a subtree with no .gitignore in play, the fallback must match exactly — proving the
    // divergence above is specifically about ignore semantics and not a broken translation.
    let dir = fixture();
    let sub = dir.path().join("sub");
    let input = json!({ "pattern": "NEEDLE", "path": sub.to_str().unwrap() });

    let good = run_tool(Some(shell_best_backend().await), "grep", input.clone()).await;
    let fallback = run_tool(Some(shell_posix_backend()), "grep", input.clone()).await;
    assert_eq!(
        good, fallback,
        "outside ignore semantics the two rungs must agree\n--- rg ---\n{good}\n--- posix ---\n{fallback}"
    );
}

// ------------------------------------------------------------- truncation

#[tokio::test]
async fn a_truncated_result_agrees_on_count_and_notice_if_not_on_membership() {
    // `LocalFs`'s parallel walk quits the instant it has `limit` matches, so *which* matches survive
    // is scheduling-dependent and documented as such. `ShellFs` scans to completion and truncates
    // afterwards. Comparing membership here would be asserting a coin flip; comparing the count and
    // the notice is the real contract.
    let dir = tempfile::tempdir().unwrap();
    for i in 0..50 {
        std::fs::write(dir.path().join(format!("f{i:03}.txt")), "NEEDLE\n").unwrap();
    }
    let input = json!({ "pattern": "NEEDLE", "path": dir.path().to_str().unwrap(), "limit": 10 });

    let l = run_tool(Some(local_backend()), "grep", input.clone()).await;
    let s = run_tool(Some(shell_best_backend().await), "grep", input.clone()).await;

    assert_eq!(
        match_lines(&l),
        10,
        "local should report exactly the limit: {l}"
    );
    assert_eq!(
        match_lines(&s),
        10,
        "shell should report exactly the limit: {s}"
    );
    for out in [&l, &s] {
        assert!(
            out.contains("match limit 10 reached"),
            "both must carry the same truncation notice: {out}"
        );
    }
}
