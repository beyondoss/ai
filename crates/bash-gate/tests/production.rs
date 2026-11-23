//! Production runs the code Verus proves: the agent's bash tool calls this crate for the limit and the
//! hand-back, and keeps no second copy of either.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

fn crates() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn production_calls_the_verified_kernel() {
    let read = |p: &str| std::fs::read_to_string(crates().join(p)).unwrap();
    let jobs = read("agent/src/tools/bash_jobs.rs");
    assert!(
        jobs.contains("bash_gate::admit_start("),
        "bash_jobs.rs no longer calls admit_start"
    );
    assert!(
        !jobs.contains("running.len() >= MAX_RUNNING_JOBS"),
        "a limit test outside the kernel"
    );
    let bash = read("agent/src/tools/bash.rs");
    assert!(
        bash.contains("bash_gate::yield_reason("),
        "bash.rs no longer calls yield_reason"
    );
    assert!(
        bash.contains("bash_gate::IDLE_YIELD_MS"),
        "IDLE_YIELD is not the kernel's"
    );
    assert!(
        !bash.contains("if quiet >= window"),
        "a silence test outside the kernel"
    );
}

/// Every decision is inside `verus!`, with no assume, admit or unlisted external body.
#[test]
fn the_decisions_are_in_the_verified_block() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"),
    )
    .unwrap();
    let open = src.find("verus! {").expect("verus! block");
    let close = src.find("} // verus!").expect("end of verus! block");
    let verified = &src[open..close];
    for decision in [
        "pub fn admit_start(",
        "pub fn yield_reason(",
        "pub proof fn lemma_a_start_stops_nothing(",
        "pub proof fn lemma_a_call_returns_by_its_ceiling(",
    ] {
        assert!(
            verified.contains(decision),
            "{decision} is not inside verus!"
        );
    }
    for banned in ["assume(", "admit()", "external_body", "verifier::external"] {
        assert!(!verified.contains(banned), "{banned} inside verus!");
    }
}
