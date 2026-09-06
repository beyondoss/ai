//! Density eval: the default agent binary must not link QuickJS.
//!
//! Idle RSS in a 768 MB guest is dominated by resident `.text`. Code Mode is a cargo feature so
//! every agent VM does not pay ~1 MB of interpreter code when `--code-mode` is never passed.
//! This test is the merge bar for that property: it runs on the density-default graph (no
//! `--features code-mode`) and fails if `JS_NewRuntime` is in the shipped binary.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[cfg(not(feature = "code-mode"))]
#[test]
fn default_agent_binary_does_not_link_quickjs() {
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let bytes = std::fs::read(bin).unwrap_or_else(|e| panic!("read {bin}: {e}"));
    let needle = b"JS_NewRuntime";
    assert!(
        !bytes.windows(needle.len()).any(|w| w == needle),
        "{bin} links QuickJS (`JS_NewRuntime` is in the image). Default/release builds must omit \
         `--features code-mode` — that is the agent-VM density contract."
    );
}
