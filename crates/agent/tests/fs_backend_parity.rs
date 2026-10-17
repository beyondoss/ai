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

/// `ShellFs` on the **`rg` rung** — the one expected to match `LocalFs` exactly. `None` when this
/// machine has no ripgrep.
///
/// Absent `rg` is handled asymmetrically on purpose. On a contributor's machine it is a skip, because
/// the POSIX-rung tests below still carry real coverage and a wall of failures for a missing optional
/// tool helps nobody. **In CI it is a hard failure**, because there the strict rung silently ceasing
/// to be tested is precisely the outcome this suite exists to prevent — and that is not theoretical:
/// GitHub's runner image ships no ripgrep, this assertion caught it, and `mise.toml` now installs it.
async fn rg_backend() -> Option<Arc<dyn FsBackend>> {
    let backend = ShellFs::connect(Arc::new(RealRunner)).await;
    if backend.capabilities().search_engine() == SearchEngine::Ripgrep {
        return Some(Arc::new(backend));
    }
    assert!(
        std::env::var("CI").is_err(),
        "CI has no `rg`, so the strict parity rung would not be tested — install ripgrep on the \
         runner (see mise.toml) rather than letting this coverage lapse"
    );
    eprintln!("skipping the ripgrep rung: no `rg` on this machine (the POSIX rung still runs)");
    None
}

/// `ShellFs` with `rg` forced absent — the fallback rung, exercised even on a box that has ripgrep.
fn shell_posix_backend() -> Arc<dyn FsBackend> {
    Arc::new(ShellFs::with_capabilities(
        Arc::new(RealRunner),
        Capabilities {
            rg: false,
            // The GNU rung: `-Z` and `-printf` available, ripgrep not. Busybox — which has neither —
            // is covered by the live suite against a real Alpine container, since faking its absence
            // here would only test the flag plumbing, not the tool it stands in for.
            grep_null: true,
            find_printf: true,
            // The chunked write fallback — what a v1 endpoint or `docker exec` without `-i` gets —
            // so every `write`/`edit` parity case covers it too. The ripgrep rung is `connect`ed and
            // probes stdin as present, so between them both write paths are diffed against `LocalFs`.
            stdin: false,
        },
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
    let Some(rg) = rg_backend().await else {
        return l;
    };
    let s = run_tool(Some(rg), tool, input.clone()).await;
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
async fn a_single_file_root_is_searched_identically_on_every_rung() {
    // Both `rg` and `grep` omit the path when handed one file, and the record parser needs it — so a
    // file-path search came back "no matches" on both shell rungs until `-H`/`--with-filename`.
    let dir = fixture();
    let file = dir.path().join("gamma.txt");
    let input = json!({ "pattern": "NEEDLE", "path": file.to_str().unwrap() });
    let out = assert_parity(&dir, input.clone()).await;
    assert!(out.contains("NEEDLE in text"), "{out}");
    let posix = run_tool(Some(shell_posix_backend()), "grep", input).await;
    assert_eq!(out, posix);
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
    // A NUL in the first 8 KB: reported by size on every backend, never decoded.
    let dir = fixture();
    let p = dir.path().join("blob.bin");
    let out = assert_tool_parity("read", json!({ "path": p.to_str().unwrap() }), &dir).await;
    assert_eq!(out, "binary file, 17 bytes — not shown");
}

fn utf16(text: &str, big_endian: bool) -> Vec<u8> {
    text.encode_utf16()
        .flat_map(|u| {
            if big_endian {
                u.to_be_bytes()
            } else {
                u.to_le_bytes()
            }
        })
        .collect()
}

#[tokio::test]
async fn read_of_an_executable_is_one_line_even_with_a_small_limit() {
    // An ELF: a header, then long runs of NULs. Decoded line by line, `limit: 5` still returned
    // kilobytes, because a binary's "lines" are arbitrarily long.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("prog");
    let mut elf = b"\x7fELF\x02\x01\x01\0".to_vec();
    elf.extend(std::iter::repeat_n(0u8, 6000));
    elf.extend(b"\n.text\n.data\n");
    std::fs::write(&p, &elf).unwrap();
    for (name, backend) in all_backends().await {
        let out = run_tool(
            Some(backend),
            "read",
            json!({ "path": p.to_str().unwrap(), "limit": 5 }),
        )
        .await;
        assert_eq!(
            out,
            format!("binary file, {} bytes — not shown", elf.len()),
            "{name}"
        );
    }
}

#[tokio::test]
async fn read_decodes_utf16_with_and_without_a_bom_on_every_backend() {
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        (
            "le-bom.txt",
            utf16("\u{feff}hello\r\nwörld\r\n", false),
            "UTF-16LE",
        ),
        (
            "be-bom.txt",
            utf16("\u{feff}hello\r\nwörld\r\n", true),
            "UTF-16BE",
        ),
        (
            "le-bare.txt",
            utf16("hello\r\nwörld\r\n", false),
            "UTF-16LE",
        ),
    ];
    for (file, bytes, label) in &cases {
        let p = dir.path().join(file);
        std::fs::write(&p, bytes).unwrap();
        for (name, backend) in all_backends().await {
            let out = run_tool(
                Some(backend),
                "read",
                json!({ "path": p.to_str().unwrap() }),
            )
            .await;
            assert!(
                out.starts_with("     1\thello\n     2\twörld\n"),
                "{name} {file}: {out:?}"
            );
            assert!(out.contains(label), "{name} {file}: {out:?}");
            assert!(
                !out.contains('\u{feff}'),
                "{name} {file}: BOM shown: {out:?}"
            );
        }
    }
}

#[tokio::test]
async fn edit_and_write_keep_a_utf16_file_utf16_on_every_backend() {
    for (name, backend) in all_backends().await {
        let dir = tempfile::tempdir().unwrap();
        for big_endian in [false, true] {
            let p = dir.path().join(format!("f-{big_endian}.ps1"));
            std::fs::write(
                &p,
                utf16("\u{feff}Write-Host one\r\nexit 0\r\n", big_endian),
            )
            .unwrap();
            let path = p.to_str().unwrap();

            run_tool(
                Some(backend.clone()),
                "edit",
                json!({ "path": path, "old_string": "one", "new_string": "two" }),
            )
            .await;
            assert_eq!(
                std::fs::read(&p).unwrap(),
                utf16("\u{feff}Write-Host two\r\nexit 0\r\n", big_endian),
                "{name} edit (big_endian={big_endian})"
            );

            // What `read` showed (no BOM) written back: still UTF-16, BOM restored.
            let out = run_tool(
                Some(backend.clone()),
                "write",
                json!({ "path": path, "content": "Write-Host three\r\n" }),
            )
            .await;
            assert!(out.contains("kept the file's UTF-16"), "{name}: {out}");
            assert_eq!(
                std::fs::read(&p).unwrap(),
                utf16("\u{feff}Write-Host three\r\n", big_endian),
                "{name} write (big_endian={big_endian})"
            );
        }
        // A new file is still plain UTF-8.
        let fresh = dir.path().join("new.txt");
        run_tool(
            Some(backend.clone()),
            "write",
            json!({ "path": fresh.to_str().unwrap(), "content": "hi" }),
        )
        .await;
        assert_eq!(std::fs::read(&fresh).unwrap(), b"hi", "{name}");
    }
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
    let Some(rg) = rg_backend().await else { return };
    let reg_s = registry_for(Some(rg));
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
    let Some(rg) = rg_backend().await else { return };
    let s = run_tool_err(Some(rg), "read", input).await;
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
        let Some(rg) = rg_backend().await else { return };
        let s = run_tool_err(Some(rg), "ls", input).await;
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

// ------------------------------------------------------------- .git

/// A repository's metadata, plus a worktree checkout whose `.git` is a *file* — each holding a
/// `NEEDLE`, so a search that walks into them shows it. The real thing: `packed-refs`, a reflog, a
/// worktree `HEAD`, the paths that filled a repository-root search before `.git` was skipped.
fn git_fixture() -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path();
    std::fs::create_dir_all(p.join(".git/logs")).unwrap();
    std::fs::create_dir_all(p.join(".git/worktrees/wt")).unwrap();
    std::fs::write(p.join(".git/packed-refs"), "abc refs/heads/NEEDLE\n").unwrap();
    std::fs::write(p.join(".git/logs/HEAD"), "commit: NEEDLE\n").unwrap();
    std::fs::write(p.join(".git/worktrees/wt/HEAD"), "ref: NEEDLE\n").unwrap();
    std::fs::create_dir(p.join("wt")).unwrap();
    std::fs::write(p.join("wt/.git"), "gitdir: NEEDLE\n").unwrap();
    std::fs::write(p.join("wt/lib.rs"), "fn NEEDLE() {}\n").unwrap();
    std::fs::write(p.join("main.rs"), "NEEDLE\n").unwrap();
    // Look-alikes that are *not* `.git` and must still be searched.
    std::fs::create_dir(p.join(".github")).unwrap();
    std::fs::write(p.join(".github/ci.yml"), "NEEDLE\n").unwrap();
    dir
}

/// Every backend this suite has — `LocalFs`, the `rg` rung when present, and the POSIX rung.
async fn all_backends() -> Vec<(&'static str, Arc<dyn FsBackend>)> {
    let mut v = vec![("local", local_backend()), ("posix", shell_posix_backend())];
    if let Some(rg) = rg_backend().await {
        v.push(("rg", rg));
    }
    v
}

#[tokio::test]
async fn a_match_deep_in_a_long_line_is_windowed_identically_on_every_backend() {
    // A lockfile/SVG-shaped single line, with the match far past the 500-character head and a
    // multi-byte neighbourhood, plus a context line that stays head-clipped.
    let dir = tempfile::tempdir().unwrap();
    let line = format!("{}é NEEDLE ü{}", "a".repeat(20_000), "b".repeat(5_000));
    let ctx = "c".repeat(900);
    std::fs::write(dir.path().join("long.txt"), format!("{ctx}\n{line}\n")).unwrap();
    let input = json!({ "pattern": "NEEDLE", "path": dir.path().to_str().unwrap(), "before": 1 });
    let mut outs = Vec::new();
    for (name, backend) in all_backends().await {
        let out = run_tool(Some(backend), "grep", input.clone()).await;
        assert!(out.contains("é NEEDLE ü"), "{name}: {out}");
        assert!(
            out.contains("columns 19756-20255 of 25010]"),
            "{name}: {out}"
        );
        outs.push((name, out));
    }
    for (name, out) in &outs[1..] {
        assert_eq!(&outs[0].1, out, "local vs {name} disagreed");
    }
}

#[tokio::test]
async fn grep_and_find_print_paths_read_can_open_on_every_backend() {
    // Relative to the tool root, not the search root: `path: "sub"` must print `sub/inner/b.txt`,
    // and `read` (same root, same backend) must open exactly that.
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("sub/inner")).unwrap();
    std::fs::write(dir.path().join("sub/inner/b.txt"), "NEEDLE\n").unwrap();
    for (name, backend) in all_backends().await {
        let reg = default_registry_with_config(&ToolConfig {
            fs_backend: Some(backend),
            root: dir.path().to_path_buf(),
            ..ToolConfig::new()
        });
        let run = |tool: &'static str, input: Value| {
            let t = tool_of(&reg, tool);
            async move { t.run(input).await.unwrap().text }
        };
        let grep = run("grep", json!({ "pattern": "NEEDLE", "path": "sub" })).await;
        assert_eq!(grep.trim_end(), "sub/inner/b.txt:1: NEEDLE", "{name}");
        let single = run(
            "grep",
            json!({ "pattern": "NEEDLE", "path": "sub/inner/b.txt" }),
        )
        .await;
        assert_eq!(single.trim_end(), "sub/inner/b.txt:1: NEEDLE", "{name}");
        let find = run("find", json!({ "pattern": "*.txt", "path": "sub" })).await;
        assert_eq!(find.trim_end(), "sub/inner/b.txt", "{name}");
        let read = run("read", json!({ "path": "sub/inner/b.txt" })).await;
        assert!(read.contains("NEEDLE"), "{name}: {read}");
    }
}

#[tokio::test]
async fn an_end_anchor_matches_crlf_lines_identically_on_every_backend() {
    // Every line of a CRLF file ends in `\r`, which sat between the text and `$`: `end$` matched
    // only the LF line. The POSIX rung runs the system `grep` (GNU here), which has no CRLF mode.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("win.txt"),
        "alpha end\r\nbeta\r\ncost $5\r\ngamma end\n",
    )
    .unwrap();
    let root = dir.path().to_str().unwrap();
    let cases = [
        ("end$", vec!["alpha end", "gamma end"]),
        ("(beta|nothing)$", vec!["beta"]),
        ("^beta$", vec!["beta"]),
        // An escaped `$` is a literal, never an anchor.
        ("\\$5$", vec!["cost $5"]),
    ];
    for (pattern, want) in cases {
        let mut outs = Vec::new();
        for (name, backend) in all_backends().await {
            let out = run_tool(
                Some(backend),
                "grep",
                json!({ "pattern": pattern, "path": root }),
            )
            .await;
            for w in &want {
                assert!(
                    out.contains(&format!(": {w}\n")),
                    "{name} {pattern:?}: {out:?}"
                );
            }
            assert_eq!(match_lines(&out), want.len(), "{name} {pattern:?}: {out:?}");
            outs.push((name, out));
        }
        for (name, out) in &outs[1..] {
            assert_eq!(&outs[0].1, out, "{pattern:?}: local vs {name} disagreed");
        }
    }
}

#[tokio::test]
async fn grep_and_find_skip_git_metadata_on_every_backend() {
    let dir = git_fixture();
    let root = dir.path().to_str().unwrap();
    for (name, backend) in all_backends().await {
        let grep = run_tool(
            Some(backend.clone()),
            "grep",
            json!({ "pattern": "NEEDLE", "path": root }),
        )
        .await;
        assert!(!grep.contains(".git/"), "{name} grep entered .git: {grep}");
        assert!(
            !grep.contains("wt/.git"),
            "{name} grep read a worktree's .git file: {grep}"
        );
        for kept in ["main.rs", "wt/lib.rs", ".github/ci.yml"] {
            assert!(grep.contains(kept), "{name} grep lost {kept}: {grep}");
        }

        for pattern in ["HEAD", "*", ".git"] {
            let find = run_tool(
                Some(backend.clone()),
                "find",
                json!({ "pattern": pattern, "path": root }),
            )
            .await;
            assert!(
                !find.lines().any(|l| l.split('/').any(|c| c == ".git")),
                "{name} find {pattern:?} listed .git: {find}"
            );
        }
    }
    // And the strict rung agrees with `LocalFs` byte for byte.
    assert_parity(&dir, json!({ "pattern": "NEEDLE" })).await;
    assert_tool_parity("find", json!({ "pattern": "*" }), &dir).await;
}

#[tokio::test]
async fn an_explicit_search_inside_git_metadata_still_searches_it() {
    let dir = git_fixture();
    let git = dir.path().join(".git");
    for (name, backend) in all_backends().await {
        let grep = run_tool(
            Some(backend.clone()),
            "grep",
            json!({ "pattern": "NEEDLE", "path": git.to_str().unwrap() }),
        )
        .await;
        assert!(grep.contains("packed-refs"), "{name}: {grep}");
        assert!(grep.contains("worktrees/wt/HEAD"), "{name}: {grep}");

        let find = run_tool(
            Some(backend.clone()),
            "find",
            json!({ "pattern": "HEAD", "path": git.to_str().unwrap() }),
        )
        .await;
        assert!(find.contains("logs/HEAD"), "{name}: {find}");

        let file = run_tool(
            Some(backend.clone()),
            "grep",
            json!({ "pattern": "NEEDLE", "path": dir.path().join("wt/.git").to_str().unwrap() }),
        )
        .await;
        assert!(file.contains("gitdir: NEEDLE"), "{name}: {file}");
    }
}

// ------------------------------------------------------- write / edit

#[tokio::test]
async fn write_then_read_round_trips_identically_across_backends() {
    // Written by one backend, read back by the same one — then the two results compared. This is the
    // test that would catch a `write` that silently mangled content on its way through base64.
    let body = "line one\nline two — with an em dash and a ünïcödé tail\n";
    let mut outs = Vec::new();
    let Some(rg) = rg_backend().await else { return };
    for backend in [local_backend(), rg] {
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
    let Some(rg) = rg_backend().await else { return };
    for backend in [local_backend(), rg] {
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
    let Some(rg) = rg_backend().await else { return };
    for backend in [local_backend(), rg] {
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
    let Some(rg) = rg_backend().await else { return };
    for backend in [local_backend(), rg] {
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
    let Some(rg) = rg_backend().await else { return };
    for backend in [local_backend(), rg] {
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
    let Some(rg) = rg_backend().await else { return };
    let s = run_tool_err(Some(rg), "edit", input).await;
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
    let Some(rg) = rg_backend().await else { return };
    let shell = registry_for(Some(rg)).definitions();
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

    let Some(rg) = rg_backend().await else { return };
    let good = run_tool(Some(rg), "grep", input.clone()).await;
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

    let Some(rg) = rg_backend().await else { return };
    let good = run_tool(Some(rg), "grep", input.clone()).await;
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
    let Some(rg) = rg_backend().await else { return };
    let s = run_tool(Some(rg), "grep", input.clone()).await;

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

/// One KiB past `edit`'s 64 MiB ceiling.
const PAST_EDIT_CAP: usize = 64 * 1024 * 1024 + 1024;

#[tokio::test]
async fn edit_refuses_a_file_over_its_size_limit_and_leaves_it_untouched_on_every_backend() {
    // Matchable text up front, a marker at the very end, zeros between. Before the size check, `edit`
    // read only the first 64 MiB, applied the edit to that prefix, wrote the prefix back, and reported
    // success: the tail, marker included, was gone.
    let mut bytes = vec![0u8; PAST_EDIT_CAP];
    let head = b"needle at the top\n";
    bytes[..head.len()].copy_from_slice(head);
    let tail = b"\nthe tail must survive\n";
    let at = bytes.len() - tail.len();
    bytes[at..].copy_from_slice(tail);

    for (name, backend) in all_backends().await {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("huge.log");
        std::fs::write(&p, &bytes).unwrap();
        let err = run_tool_err(
            Some(backend),
            "edit",
            json!({ "path": p.to_str().unwrap(), "old_string": "needle", "new_string": "pin" }),
        )
        .await;
        assert!(
            err.contains(&PAST_EDIT_CAP.to_string()) && err.contains("67108864"),
            "{name}: the refusal names the size and the limit: {err}"
        );
        assert!(err.contains("bash"), "{name}: and says what to use: {err}");
        assert!(
            std::fs::read(&p).unwrap() == bytes,
            "{name}: a refused edit must leave the file byte-identical"
        );
    }
}

#[tokio::test]
async fn edit_keeps_every_byte_of_a_file_larger_than_one_command_output_on_every_backend() {
    // 1 MiB: well past what one command's captured stdout holds (128 KiB head + 128 KiB tail). A
    // single-command read kept the head and the tail, glued them together, and `edit` wrote that back.
    let mut text = String::from("needle at the top\n");
    let mut i = 0u32;
    while text.len() < 1024 * 1024 {
        text.push_str(&format!("line {i:08} of a file bigger than one capture\n"));
        i += 1;
    }
    let expected = text.replacen("needle", "pin", 1);

    for (name, backend) in all_backends().await {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big.txt");
        std::fs::write(&p, &text).unwrap();
        run_tool(
            Some(backend),
            "edit",
            json!({ "path": p.to_str().unwrap(), "old_string": "needle", "new_string": "pin" }),
        )
        .await;
        let after = std::fs::read_to_string(&p).unwrap();
        assert!(
            after == expected,
            "{name}: edit changed more than the match ({} bytes before, {} after)",
            text.len(),
            after.len()
        );
    }
}
