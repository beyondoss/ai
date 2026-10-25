//! `agent settings` must say when a project tier overrides what it just printed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use common::agent_cmd;

fn setup() -> (tempfile::TempDir, tempfile::TempDir) {
    let home = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project.path().join(".beyond")).unwrap();
    std::fs::write(
        project.path().join(".beyond/settings.json"),
        r#"{"default_model":"project-model"}"#,
    )
    .unwrap();
    (home, project)
}

fn trust(home: &tempfile::TempDir, project: &tempfile::TempDir) {
    let out = agent_cmd()
        .args(["trust", project.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(out.status.success());
}

#[test]
fn clear_model_names_the_project_file_that_still_sets_it() {
    let (home, project) = setup();
    trust(&home, &project);
    agent_cmd()
        .args(["settings", "--model", "global-model"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    let out = agent_cmd()
        .args(["settings", "--clear-model"])
        .env("HOME", home.path())
        .current_dir(project.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{stdout}");
    let expected = format!(
        "default_model: (not set) \u{2014} overridden by {}: project-model",
        project.path().join(".beyond/settings.json").display()
    );
    assert!(stdout.contains(&expected), "{stdout}");
}

#[test]
fn plain_show_names_the_override_only_for_a_trusted_project() {
    let (home, project) = setup();
    let show = |home: &tempfile::TempDir| {
        let out = agent_cmd()
            .arg("settings")
            .env("HOME", home.path())
            .current_dir(project.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    assert!(!show(&home).contains("overridden by"));
    trust(&home, &project);
    assert!(show(&home).contains("overridden by"));
}
