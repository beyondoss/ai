//! `--tool-tuning edit-schema=…`, the edit tool's A/B surfaces, and the filler-pair rule every
//! surface shares. Each case below is a shape the longctx eval actually saw GPT models send (see
//! "Edit tool surface" in ARCHITECTURE.md).

use serde_json::{Value, json};

use super::*;

fn tmp(contents: &str) -> tempfile::NamedTempFile {
    use std::io::Write as _;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    f
}

fn tools(schema: EditSchema) -> Vec<Edit> {
    Edit::default().with_options(EditOptions { schema })
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

fn props(t: &Edit) -> Vec<String> {
    let s = t.input_schema();
    let mut p: Vec<String> = s["properties"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    p.sort();
    p
}

fn required(t: &Edit) -> Vec<String> {
    let mut r: Vec<String> = t.input_schema()["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    r.sort();
    r
}

#[test]
fn combined_schema_is_unchanged() {
    let t = tools(EditSchema::Combined);
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].name(), "edit");
    assert_eq!(
        props(&t[0]),
        ["edits", "new_string", "old_string", "path", "replace_all"]
    );
    assert_eq!(required(&t[0]), ["path"]);
}

#[test]
fn edits_only_schema_advertises_no_top_level_pair() {
    let t = tools(EditSchema::EditsOnly);
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].name(), "edit");
    assert_eq!(props(&t[0]), ["edits", "path"]);
    assert_eq!(required(&t[0]), ["edits", "path"]);
}

#[test]
fn split_schema_is_a_single_edit_plus_multi_edit() {
    let t = tools(EditSchema::Split);
    let names: Vec<&str> = t.iter().map(|t| t.name()).collect();
    assert_eq!(names, ["edit", MULTI_EDIT]);
    assert_eq!(
        props(&t[0]),
        ["new_string", "old_string", "path", "replace_all"]
    );
    assert_eq!(required(&t[0]), ["new_string", "old_string", "path"]);
    assert_eq!(props(&t[1]), ["edits", "path"]);
    assert!(t[0].description().contains("multi_edit"));
    assert!(t[1].description().contains("`edit`"));
}

#[tokio::test]
async fn every_surface_accepts_the_other_shape_rather_than_failing_on_schema() {
    // A model that ignores the advertised shape still gets its edit applied.
    for schema in [EditSchema::EditsOnly, EditSchema::Split] {
        for t in tools(schema) {
            let (r, after) = run_on(
                &t,
                "alpha",
                json!({ "old_string": "alpha", "new_string": "A" }),
            )
            .await;
            r.unwrap();
            assert_eq!(after, "A");
            let (r, after) = run_on(
                &t,
                "alpha",
                json!({ "edits": [{ "old_string": "alpha", "new_string": "A" }] }),
            )
            .await;
            r.unwrap();
            assert_eq!(after, "A");
        }
    }
}

#[test]
fn multi_edit_declares_its_write_target_so_dispatch_serializes_it() {
    let t = tools(EditSchema::Split);
    assert!(t[1].write_target(&json!({ "path": "a.txt" })).is_some());
}

/// Under the `json-key-order` feature serde_json keeps insertion order, so the schema reaches the wire
/// with `path` first and `old_string` before `new_string` — the order the source writes them in.
#[cfg(feature = "json-key-order")]
#[test]
fn json_key_order_feature_serializes_properties_in_source_order() {
    let s = serde_json::to_string(&Edit::default().input_schema()).unwrap();
    let pos = |k: &str| s.find(&format!("\"{k}\"")).unwrap();
    assert!(pos("path") < pos("edits"));
    assert!(pos("old_string") < pos("new_string"));
}

/// Without it (the default build) serde_json sorts keys: what every eval run so far was sent.
#[cfg(not(feature = "json-key-order"))]
#[test]
fn default_build_serializes_schema_keys_alphabetically() {
    let s = serde_json::to_string(&Edit::default().input_schema()).unwrap();
    let pos = |k: &str| s.find(&format!("\"{k}\"")).unwrap();
    assert!(pos("edits") < pos("path"));
    assert!(pos("new_string") < pos("old_string"));
}
