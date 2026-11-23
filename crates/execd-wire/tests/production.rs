//! Production runs the code Verus proves: execd calls this crate for the heartbeat wait, the one-shot's
//! first answer and a job's signal scope, and no other source carries a second copy of them.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

fn crates() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
fn production_calls_the_verified_kernel() {
    let read = |p: &str| std::fs::read_to_string(crates().join(p)).unwrap();
    assert!(
        read("execd/src/lib.rs").contains("pub use execd_wire as wire_kernel;"),
        "execd::wire_kernel is the verified crate, not a copy"
    );
    let exec = read("execd/src/exec.rs");
    for needle in ["wire::quiet_wait(", "wire::oneshot_mode("] {
        assert!(exec.contains(needle), "exec.rs no longer calls {needle}");
    }
    let proc = read("execd/src/proc.rs");
    for needle in ["wire::is_member(", "wire::group_reaches("] {
        assert!(proc.contains(needle), "proc.rs no longer calls {needle}");
    }
    // The beat is the kernel's constant, not a second number.
    assert!(
        read("execd/src/lib.rs").contains("execd_wire::HEARTBEAT_MS"),
        "HEARTBEAT_INTERVAL is not the kernel's"
    );
    // The shell keeps no copy of the decisions: no `due = last + heartbeat`, no group-or-mark test.
    assert!(
        !exec.contains("self.last + self.heartbeat"),
        "a heartbeat due time outside the kernel"
    );
    assert!(
        !proc.contains("(pgid > 1 && pg == pgid) || has_mark"),
        "a membership test outside the kernel"
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
        "pub fn quiet_wait(",
        "pub fn oneshot_mode(",
        "pub fn is_member(",
        "pub fn group_reaches(",
        "pub proof fn lemma_never_silent_past_a_beat(",
        "pub proof fn lemma_silence_is_carried(",
        "pub proof fn lemma_every_member_is_signalled(",
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
