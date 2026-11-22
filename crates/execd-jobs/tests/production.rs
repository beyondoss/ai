//! Production runs the code Verus proves: execd's job table calls this crate's functions, the
//! decisions are written inside `verus!`, and no other source carries a second copy of them.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

fn crates() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn production_calls_the_verified_kernel() {
    let read = |p: &str| std::fs::read_to_string(crates().join(p)).unwrap();
    assert!(
        read("execd/src/lib.rs").contains("pub use execd_jobs as job_kernel;"),
        "execd::job_kernel is the verified crate, not a copy"
    );
    assert!(
        !crates().join("execd/src/job_kernel.rs").exists(),
        "a second job_kernel in execd"
    );
    let jobs = read("execd/src/jobs.rs");
    for needle in [
        "kernel::step(",
        "kernel::window(",
        "kernel::follow(",
        "kernel::bound(",
        "kernel::State::new(",
        ".admit(&key)",
    ] {
        assert!(jobs.contains(needle), "jobs.rs no longer calls {needle}");
    }
}

/// Every decision is inside `verus!`: the verified block holds each function, and the code after
/// it is only storage (`HashMap`, string comparison) and accessors that reshape a field.
#[test]
fn the_decisions_are_in_the_verified_block() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"),
    )
    .unwrap();
    let open = src.find("verus! {").expect("verus! block");
    let close = src.find("} // verus!").expect("end of verus! block");
    let verified = &src[open..close];
    let storage = &src[close..];
    for decision in [
        "pub fn step(",
        "pub fn window(",
        "pub fn follow(",
        "pub fn bound<",
        "pub fn admit(",
        "pub fn visible(",
        "pub proof fn lemma_end_progress(",
        "pub proof fn lemma_follow_progress(",
    ] {
        assert!(
            verified.contains(decision),
            "{decision} is not inside verus!"
        );
    }
    for banned in ["assume(", "admit()", "external_body", "verifier::external"] {
        assert!(!verified.contains(banned), "{banned} inside verus!");
    }
    for stray in [
        "fn step(",
        "fn window(",
        "fn follow(",
        "fn bound<",
        "Ev::",
        "Phase::",
    ] {
        assert!(
            !storage.contains(stray),
            "{stray} outside verus!: a decision the proof does not see"
        );
    }
}

/// No second copy of the kernel's state machine anywhere else in the workspace.
#[test]
fn no_second_copy() {
    let mut copies = Vec::new();
    let mut dirs = vec![crates()];
    while let Some(d) = dirs.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                if !matches!(
                    p.file_name().and_then(|n| n.to_str()),
                    Some("target" | "node_modules" | ".git")
                ) {
                    dirs.push(p);
                }
            } else if p.extension().is_some_and(|x| x == "rs")
                && !p.starts_with(crates().join("execd-jobs"))
                && std::fs::read_to_string(&p)
                    .is_ok_and(|s| s.contains("shell_told: bool") && s.contains("Exited {"))
            {
                copies.push(p);
            }
        }
    }
    assert!(
        copies.is_empty(),
        "a second copy of the job kernel: {copies:?}"
    );
}
