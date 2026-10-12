//! What a failed match says about itself. None of these apply anything: they only tell the model
//! where its text actually is.

use serde_json::json;

use super::*;

fn files(pairs: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (name, body) in pairs {
        std::fs::write(dir.path().join(name), body).unwrap();
    }
    dir
}

fn invalid(r: Result<ToolOutput, ToolError>) -> String {
    match r {
        Err(ToolError::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[tokio::test]
async fn not_found_names_the_other_file_in_the_call_that_holds_the_text() {
    // The cross-file batch shape: the model sent b.py's text under a.py's entry. b.py's own entry
    // still lands, and a.py's failure names b.py.
    let dir = files(&[("a.py", "alpha = 1\n"), ("b.py", "beta = 2\ngamma = 3\n")]);
    let msg = Edit::new(dir.path())
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "beta = 2", "new_string": "beta = 20" },
            { "path": "b.py", "old_string": "gamma", "new_string": "GAMMA" },
        ]}))
        .await
        .unwrap()
        .text;
    let b = dir.path().join("b.py");
    assert!(
        msg.contains(&format!("That text is in {}", b.display())),
        "{msg}"
    );
    assert!(
        !msg.contains("Closest match"),
        "the file is named instead: {msg}"
    );
    assert_eq!(
        std::fs::read_to_string(&b).unwrap(),
        "beta = 2\nGAMMA = 3\n"
    );
}

#[tokio::test]
async fn not_found_never_names_a_file_whose_text_this_call_replaced() {
    // b.py held the text when the call was planned, but b.py's own entry rewrites it: b.py is
    // searched as it will be written, so it is not named.
    let dir = files(&[("a.py", "alpha = 1\n"), ("b.py", "beta = 2\n")]);
    let msg = Edit::new(dir.path())
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "beta = 2", "new_string": "beta = 20" },
            { "path": "b.py", "old_string": "beta", "new_string": "BETA" },
        ]}))
        .await
        .unwrap()
        .text;
    assert!(!msg.contains("That text is in"), "{msg}");
    assert!(msg.contains("FAILED a.py (not edited):"), "{msg}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("b.py")).unwrap(),
        "BETA = 2\n"
    );
}

#[tokio::test]
async fn not_found_names_a_file_the_model_read_that_holds_the_text() {
    // A single-file call: the batch has no other file, so the hint comes from what `read` touched.
    let dir = files(&[
        ("a.py", "alpha = 1\n"),
        ("b.py", "def f():\n    return compute(x, y)\n"),
    ]);
    let touched = Arc::new(crate::tools::TouchedFiles::default());
    let read = crate::tools::read::Read::new(dir.path()).with_touched(touched.clone());
    read.run(json!({ "path": "b.py" })).await.unwrap();
    let msg = invalid(
        Edit::new(dir.path())
            .with_touched(touched)
            .run(json!({ "path": "a.py", "old_string": "return compute(x, y)", "new_string": "return 0" }))
            .await,
    );
    assert!(
        msg.contains(&format!(
            "That text is in {}",
            dir.path().join("b.py").display()
        )),
        "{msg}"
    );
}

#[tokio::test]
async fn an_untouched_file_is_never_searched() {
    let dir = files(&[("a.py", "alpha = 1\n"), ("b.py", "beta = 2\n")]);
    let msg = invalid(
        Edit::new(dir.path())
            .run(json!({ "path": "a.py", "old_string": "beta = 2", "new_string": "x" }))
            .await,
    );
    assert!(!msg.contains("That text is in"), "{msg}");
}

#[tokio::test]
async fn not_found_says_already_applied_when_the_new_string_is_present() {
    let dir = files(&[(
        "a.py",
        "import os\n\ndef run(path, *, strict=True):\n    pass\n",
    )]);
    let msg = invalid(
        Edit::new(dir.path())
            .run(json!({
                "path": "a.py",
                "old_string": "def run(path):",
                "new_string": "def run(path, *, strict=True):",
            }))
            .await,
    );
    assert!(
        msg.ends_with(
            "`new_string` is already present at line 3: this edit looks already applied."
        ),
        "{msg}"
    );
}

#[tokio::test]
async fn not_found_shows_the_closest_lines_with_their_numbers() {
    let body = "import os\n\nclass Parser:\n    def parse(self, text):\n        tokens = text.split()\n        return [t.lower() for t in tokens]\n\n    def other(self):\n        return None\n";
    let dir = files(&[("a.py", body)]);
    let msg = invalid(
        Edit::new(dir.path())
            .run(json!({
                "path": "a.py",
                // misremembered: `split(",")` and `t.upper()`
                "old_string": "        tokens = text.split(\",\")\n        return [t.upper() for t in tokens]",
                "new_string": "        return []",
            }))
            .await,
    );
    assert!(
        msg.contains(
            "Closest match in the file (lines 5-6):\n     5\t        tokens = text.split()\n     6\t        return [t.lower() for t in tokens]"
        ),
        "{msg}"
    );
}

#[tokio::test]
async fn the_closest_lines_snippet_is_bounded() {
    let body: String = (1..=60)
        .map(|i| format!("value_{i} = compute({i})\n"))
        .collect();
    let old: String = (10..=40)
        .map(|i| format!("value_{i} = compute({i}, x)\n"))
        .collect();
    let dir = files(&[("a.py", &body)]);
    let msg = invalid(
        Edit::new(dir.path())
            .run(json!({ "path": "a.py", "old_string": old, "new_string": "" }))
            .await,
    );
    let snippet = msg.split("Closest match").nth(1).expect(&msg);
    let shown = snippet.lines().filter(|l| l.contains('\t')).count();
    assert_eq!(shown, SNIPPET_LINES, "{msg}");
    assert!(snippet.trim_end().ends_with('…'), "{msg}");
    assert!(msg.len() <= HINT_MAX_BYTES * 2, "{} bytes", msg.len());
}

#[tokio::test]
async fn not_found_with_nothing_similar_adds_no_snippet() {
    let dir = files(&[("a.py", "alpha = 1\n")]);
    let msg = invalid(
        Edit::new(dir.path())
            .run(json!({ "path": "a.py", "old_string": "zzzz qqqq", "new_string": "x" }))
            .await,
    );
    assert_eq!(
        msg,
        format!(
            "`old_string` not found: \"zzzz qqqq\" in {}",
            dir.path().join("a.py").display()
        )
    );
}

#[tokio::test]
async fn not_unique_lists_each_match_with_its_enclosing_definition() {
    let body = "class Reader:\n    def load(self):\n        self.reset()\n        return 1\n\nclass Writer:\n    def save(self):\n        self.reset()\n        return 2\n\nself.reset()\n";
    let dir = files(&[("a.py", body)]);
    let msg = invalid(
        Edit::new(dir.path())
            .run(json!({ "path": "a.py", "old_string": "self.reset()", "new_string": "pass" }))
            .await,
    );
    assert!(msg.contains("is not unique (3 matches)"), "{msg}");
    assert!(
        msg.ends_with(
            "Matches at line 3 (in `def load(self):`), line 8 (in `def save(self):`), line 11."
        ),
        "{msg}"
    );
}

#[tokio::test]
async fn not_unique_finds_rust_scopes_and_caps_the_list() {
    let mut body = String::from("impl Thing {\n");
    for i in 0..10 {
        body.push_str(&format!(
            "    pub fn f{i}(&self) {{\n        tick();\n    }}\n"
        ));
    }
    body.push_str("}\n");
    let dir = files(&[("a.rs", &body)]);
    let msg = invalid(
        Edit::new(dir.path())
            .run(json!({ "path": "a.rs", "old_string": "tick();", "new_string": "tock();" }))
            .await,
    );
    assert!(msg.contains("line 3 (in `pub fn f0(&self) {`)"), "{msg}");
    assert!(msg.ends_with(" and 2 more."), "{msg}");
}

#[tokio::test]
async fn a_long_old_string_is_echoed_capped() {
    let long = "x".repeat(5000);
    let dir = files(&[("a.py", "alpha\n")]);
    let msg = invalid(
        Edit::new(dir.path())
            .run(json!({ "path": "a.py", "old_string": long, "new_string": "y" }))
            .await,
    );
    assert!(msg.contains("… (5000 chars)"), "{msg}");
    assert!(msg.len() < 600, "{} bytes: {msg}", msg.len());
}

#[test]
fn enclosing_scope_needs_a_less_indented_definition() {
    let lines = ["def a():", "    x = 1", "y = 2", "class B:", "    z = 3"];
    assert_eq!(enclosing_scope(&lines, 1), Some("def a():"));
    assert_eq!(
        enclosing_scope(&lines, 2),
        None,
        "top level has no enclosing scope"
    );
    assert_eq!(enclosing_scope(&lines, 4), Some("class B:"));
}
