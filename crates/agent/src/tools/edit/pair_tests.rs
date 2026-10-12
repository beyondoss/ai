//! The filler-pair rule: what `edit` does with a top-level `old_string`/`new_string` sent beside
//! `edits[]`. Each case below is a shape the longctx eval actually saw GPT models send (see "Edit tool
//! surface" in ARCHITECTURE.md).

use serde_json::{Value, json};

use super::*;

fn tmp(contents: &str) -> tempfile::NamedTempFile {
    use std::io::Write as _;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    f
}

async fn run_on(
    tool: &Edit,
    contents: &str,
    mut input: Value,
) -> (Result<ToolOutput, ToolError>, String) {
    let f = tmp(contents);
    input["path"] = json!(f.path().to_str().unwrap());
    let r = tool.run(input).await;
    (r, std::fs::read_to_string(f.path()).unwrap())
}

fn invalid(r: Result<ToolOutput, ToolError>) -> String {
    match r {
        Err(ToolError::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[tokio::test]
async fn drops_an_empty_pair() {
    let (r, after) = run_on(
        &Edit::default(),
        "alpha beta",
        json!({ "edits": [{ "old_string": "alpha", "new_string": "ALPHA" }], "old_string": "", "new_string": "" }),
    )
    .await;
    r.unwrap();
    assert_eq!(after, "ALPHA beta");
}

#[tokio::test]
async fn drops_a_no_op_pair_even_one_that_would_not_match() {
    // The anchor-to-itself workaround (`"import x"` → `"import x"`), and the placeholder that matches
    // nothing (`"unused"` → `"unused"`): neither can change the file, so neither may fail the call.
    for anchor in ["beta", "unused", "a"] {
        let (r, after) = run_on(
            &Edit::default(),
            "alpha beta",
            json!({ "edits": [{ "old_string": "alpha", "new_string": "ALPHA" }], "old_string": anchor, "new_string": anchor }),
        )
        .await;
        r.unwrap_or_else(|e| panic!("anchor {anchor:?}: {e:?}"));
        assert_eq!(after, "ALPHA beta");
    }
}

#[tokio::test]
async fn drops_a_pair_that_duplicates_an_entry() {
    // Appended, it would read as two overlapping edits of the same span.
    let (r, after) = run_on(
        &Edit::default(),
        "alpha beta",
        json!({ "edits": [{ "old_string": "alpha", "new_string": "ALPHA" }], "old_string": "alpha", "new_string": "ALPHA" }),
    )
    .await;
    r.unwrap();
    assert_eq!(after, "ALPHA beta");
}

#[tokio::test]
async fn still_applies_a_pair_that_is_a_real_different_edit() {
    // Sol used the pair as a genuine extra edit in 27% of its calls; dropping it would lose work.
    let (r, after) = run_on(
        &Edit::default(),
        "alpha beta",
        json!({ "edits": [{ "old_string": "alpha", "new_string": "ALPHA" }], "old_string": "beta", "new_string": "BETA" }),
    )
    .await;
    r.unwrap();
    assert_eq!(after, "ALPHA BETA");
}

#[tokio::test]
async fn still_rejects_a_pair_that_conflicts_with_an_entry() {
    // Same span, different replacement: which one the model meant is unknowable, so it stays an error.
    let (r, after) = run_on(
        &Edit::default(),
        "alpha beta",
        json!({ "edits": [{ "old_string": "alpha", "new_string": "ALPHA" }], "old_string": "alpha", "new_string": "omega" }),
    )
    .await;
    assert!(invalid(r).contains("overlap"));
    assert_eq!(after, "alpha beta");
}

#[tokio::test]
async fn keeps_an_empty_old_string_with_a_real_replacement_an_error() {
    // Not filler: the model wrote something it wanted inserted, and there is no anchor to put it at.
    let (r, _) = run_on(
        &Edit::default(),
        "alpha beta",
        json!({ "edits": [{ "old_string": "alpha", "new_string": "ALPHA" }], "old_string": "", "new_string": "gamma" }),
    )
    .await;
    assert!(invalid(r).contains("`old_string` is empty"));
}

#[tokio::test]
async fn uses_the_pair_when_edits_is_an_empty_array() {
    let (r, after) = run_on(
        &Edit::default(),
        "alpha beta",
        json!({ "edits": [], "old_string": "beta", "new_string": "BETA" }),
    )
    .await;
    r.unwrap();
    assert_eq!(after, "alpha BETA");
    // …but an empty array beside an empty pair is still nothing to do.
    let (r, _) = run_on(
        &Edit::default(),
        "alpha",
        json!({ "edits": [], "old_string": "", "new_string": "" }),
    )
    .await;
    assert!(invalid(r).contains("`edits` is empty"));
}

#[tokio::test]
async fn applies_replace_all_to_the_one_entry_left_after_dropping_filler() {
    // Sol set `replace_all: true` meaning its single edits[] entry; the filler pair made it "two edits".
    let (r, after) = run_on(
        &Edit::default(),
        "x x x",
        json!({ "edits": [{ "old_string": "x", "new_string": "y" }], "old_string": "", "new_string": "", "replace_all": true }),
    )
    .await;
    r.unwrap();
    assert_eq!(after, "y y y");
}
