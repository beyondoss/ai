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

/// N2: a path with a newline in it spreads the engine's one-line error over two lines, and the D4
/// classifier then reads the missing root as a bad regex.
#[test]
fn n2_missing_search_root_with_a_newline_is_a_bad_regex() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("no\nwhere");
    let err = env::rt().block_on(fs().search(&query(&root))).unwrap_err();
    assert!(
        matches!(err, FsError::InvalidQuery(_)),
        "N2 is fixed: delete this test and the known(\"N2\") call; got {err}"
    );
}

/// N3: a search root below a regular file ("Not a directory") is not among the OS errors the D4
/// classifier knows, so it is reported as a bad regex.
#[test]
fn n3_search_root_below_a_file_is_a_bad_regex() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file"), b"needle").unwrap();
    let root = dir.path().join("file/sub");
    let err = env::rt().block_on(fs().search(&query(&root))).unwrap_err();
    assert!(
        matches!(err, FsError::InvalidQuery(_)),
        "N3 is fixed: delete this test and the known(\"N3\") call; got {err}"
    );
}

/// N4: the write script checks the temp's size by reading it back (`wc -c < tmp`), and the temp has
/// the target's mode, so a file its owner may write but not read cannot be written.
#[test]
fn n4_a_write_only_file_cannot_be_written() {
    if env::truth_is_root() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("w");
    std::fs::write(&p, b"old").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o200)).unwrap();
    let r = env::rt().block_on(fs().write_bytes(&p, b"new".to_vec()));
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(
        r.is_err(),
        "N4 is fixed: delete this test and the known(\"N4\") call"
    );
    assert_eq!(std::fs::read(&p).unwrap(), b"old");
}
