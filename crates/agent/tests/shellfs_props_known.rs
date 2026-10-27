//! Open bugs the property tests found, pinned one by one.
//!
//! The properties tolerate an open bug (they print `KNOWN OPEN BUG Nx` and move on) so that one
//! defect does not hide every other. Each test below reproduces one of them deterministically and
//! asserts it is **still there**: when its fix lands the assertion fails, which is the cue to delete
//! the test and the matching `known("Nx", ..)` call in `shellfs_props_env/mod.rs`, so the property
//! starts enforcing it. The same cases are in `sdk/conformance/scenarios/files.json`, marked `xfail`
//! for the reference driver, so every other client has to pass them.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use crate::shellfs_props_env as env;
use beyond_agent::tools::fs::shell::ShellFs;
use beyond_agent::tools::fs::{FsBackend, FsError, SearchQuery};

fn fs() -> ShellFs {
    env::rt()
        .block_on(env::connect(env::http_runner(None)))
        .expect("loopback execd")
}

fn query(root: &Path) -> SearchQuery {
    SearchQuery {
        pattern: "needle".into(),
        literal: true,
        ignore_case: false,
        glob: None,
        root: root.to_path_buf(),
        limit: 10,
        before: 0,
        after: 0,
    }
}

/// N1 (fixed): a symlink to a directory lists what it points to, under the link's path.
#[test]
fn n1_list_of_a_symlinked_directory_lists_the_target() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("real")).unwrap();
    std::fs::write(dir.path().join("real/f"), b"x").unwrap();
    std::os::unix::fs::symlink("real", dir.path().join("link")).unwrap();
    let fs = fs();
    let link = dir.path().join("link");
    let st = env::rt().block_on(fs.stat(&link)).unwrap().unwrap();
    assert_eq!(format!("{:?}", st.kind), "Dir");
    let entries = env::rt().block_on(fs.list_dir(&link, 100, true)).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "f");
}

/// N3 (fixed): a search root below a regular file is `not_a_directory`, not a bad regex.
#[test]
fn n3_search_root_below_a_file_is_not_a_directory() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file"), b"needle").unwrap();
    let root = dir.path().join("file/sub");
    let err = env::rt().block_on(fs().search(&query(&root))).unwrap_err();
    assert!(err.to_string().contains("Not a directory"), "{err}");
    assert!(!matches!(err, FsError::InvalidQuery(_)), "{err}");
}

/// N4 (fixed): a file its owner may write but not read is replaced, and keeps its mode.
#[test]
fn n4_a_write_only_file_can_be_written() {
    if env::truth_is_root() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("w");
    std::fs::write(&p, b"old").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o200)).unwrap();
    let r = env::rt().block_on(fs().write_bytes(&p, b"new".to_vec()));
    let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o7777;
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
    r.unwrap();
    assert_eq!(mode, 0o200);
    assert_eq!(std::fs::read(&p).unwrap(), b"new");
}

/// N5 (fixed): the temp's name does not depend on the target's, so a name up to NAME_MAX can be
/// written, existing or new, and no temp survives.
#[test]
fn n5_a_name_near_name_max_can_be_written() {
    let dir = tempfile::tempdir().unwrap();
    let fs = fs();
    for len in [229usize, 250, 255] {
        let p = dir.path().join("n".repeat(len));
        for round in 0..2 {
            let body = format!("v{round}").into_bytes();
            env::rt()
                .block_on(fs.write_bytes(&p, body.clone()))
                .unwrap_or_else(|e| panic!("{len}-byte name, round {round}: {e}"));
            assert_eq!(std::fs::read(&p).unwrap(), body);
        }
    }
    let temps: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .filter(|n| env::truth::looks_like_temp(n))
        .collect();
    assert!(temps.is_empty(), "{temps:?}");
}
