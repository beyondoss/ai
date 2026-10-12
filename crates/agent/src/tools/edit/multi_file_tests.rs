//! Per-entry `path`: one `edit` call across several files, planned in full before any write.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use serde_json::json;

use super::*;
use crate::tools::fs::{
    DirEntry, FsError, GlobOutcome, GlobQuery, Meta, PathWorld, SearchOutcome, SearchQuery,
};

fn files(pairs: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (name, body) in pairs {
        std::fs::write(dir.path().join(name), body).unwrap();
    }
    dir
}

fn read(dir: &tempfile::TempDir, name: &str) -> String {
    std::fs::read_to_string(dir.path().join(name)).unwrap()
}

fn message(r: Result<ToolOutput, ToolError>) -> String {
    match r {
        Ok(o) => panic!("expected an error, got {:?}", o.text),
        Err(ToolError::InvalidInput(m) | ToolError::Execution(m)) => m,
        Err(e) => panic!("unexpected error {e:?}"),
    }
}

#[tokio::test]
async fn entries_naming_their_own_path_edit_several_files_in_one_call() {
    let dir = files(&[("a.py", "alpha = 1\n"), ("b.py", "beta = 2\n")]);
    let out = Edit::new(dir.path())
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "alpha = 1", "new_string": "alpha = 10" },
            { "path": "b.py", "old_string": "beta = 2", "new_string": "beta = 20" },
        ]}))
        .await
        .unwrap();
    assert_eq!(read(&dir, "a.py"), "alpha = 10\n");
    assert_eq!(read(&dir, "b.py"), "beta = 20\n");
    let a = dir.path().join("a.py");
    let b = dir.path().join("b.py");
    assert_eq!(
        out.text,
        format!(
            "edited {} (1 replacement)\nedited {} (1 replacement)",
            a.display(),
            b.display()
        )
    );
}

#[tokio::test]
async fn a_single_file_call_reads_exactly_as_it_always_has() {
    let dir = files(&[("a.py", "x = 1\ny = 2\n")]);
    let out = Edit::new(dir.path())
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "x = 1", "new_string": "x = 3" },
            { "old_string": "y = 2", "new_string": "y = 4" },
        ]}))
        .await
        .unwrap();
    let a = dir.path().join("a.py");
    assert_eq!(out.text, format!("edited {} (2 replacements)", a.display()));
}

#[tokio::test]
async fn a_blank_entry_path_means_the_top_level_one() {
    // Models that fill every advertised property send `"path": ""` for "the default".
    let dir = files(&[("a.py", "one two\n")]);
    Edit::new(dir.path())
        .run(json!({ "path": "a.py", "edits": [
            { "path": "", "old_string": "one", "new_string": "1" },
            { "path": "  ", "old_string": "two", "new_string": "2" },
        ]}))
        .await
        .unwrap();
    assert_eq!(read(&dir, "a.py"), "1 2\n");
}

#[tokio::test]
async fn one_failing_file_writes_no_file_and_reports_each_one() {
    let dir = files(&[
        ("a.py", "alpha = 1\n"),
        ("b.py", "beta = 2\n"),
        ("c.py", "gamma = 3\n"),
    ]);
    let r = Edit::new(dir.path())
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "alpha = 1", "new_string": "alpha = 10" },
            { "path": "b.py", "old_string": "beta = 999", "new_string": "beta = 20" },
            { "path": "c.py", "old_string": "gamma = 3", "new_string": "gamma = 30" },
        ]}))
        .await;
    assert!(matches!(r, Err(ToolError::InvalidInput(_))), "{r:?}");
    let msg = message(r);
    assert!(msg.starts_with("no file was edited"), "{msg}");
    assert!(msg.contains("not found in edits[1]"), "{msg}");
    assert!(msg.contains("b.py"), "{msg}");
    assert!(
        msg.contains("were valid but not written") && msg.contains("a.py") && msg.contains("c.py"),
        "{msg}"
    );
    assert_eq!(read(&dir, "a.py"), "alpha = 1\n", "a.py must be untouched");
    assert_eq!(read(&dir, "c.py"), "gamma = 3\n", "c.py must be untouched");
}

#[tokio::test]
async fn a_missing_file_in_the_batch_writes_nothing() {
    let dir = files(&[("a.py", "alpha = 1\n")]);
    let r = Edit::new(dir.path())
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "alpha = 1", "new_string": "alpha = 10" },
            { "path": "missing.py", "old_string": "x", "new_string": "y" },
        ]}))
        .await;
    // A file that cannot be read is an execution failure, not a bad argument.
    assert!(matches!(r, Err(ToolError::Execution(_))), "{r:?}");
    assert!(message(r).contains("missing.py"));
    assert_eq!(read(&dir, "a.py"), "alpha = 1\n");
}

#[tokio::test]
async fn two_spellings_of_one_file_are_planned_together() {
    let dir = files(&[("a.py", "one two three\n")]);
    let edit = Edit::new(dir.path());
    // Overlapping entries under different spellings are caught by the overlap check, not written twice.
    let r = edit
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "one two", "new_string": "X" },
            { "path": "./a.py", "old_string": "two three", "new_string": "Y" },
        ]}))
        .await;
    let msg = message(r);
    assert!(msg.contains("edits[0] and edits[1] overlap"), "{msg}");
    // Disjoint ones land in one write, counted together.
    let out = edit
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "one", "new_string": "1" },
            { "path": "./a.py", "old_string": "three", "new_string": "3" },
        ]}))
        .await
        .unwrap();
    assert!(out.text.ends_with("(2 replacements)"), "{}", out.text);
    assert_eq!(read(&dir, "a.py"), "1 two 3\n");
}

#[tokio::test]
async fn a_filler_pair_copying_another_files_entry_is_dropped() {
    // #235's rule survives routing: the pair copies the b.py entry's text, so it is filler — it must
    // not be re-applied to the top-level a.py (where it would fail "not found").
    let dir = files(&[("a.py", "alpha\n"), ("b.py", "beta\n")]);
    Edit::new(dir.path())
        .run(json!({
            "path": "a.py",
            "edits": [
                { "old_string": "alpha", "new_string": "ALPHA" },
                { "path": "b.py", "old_string": "beta", "new_string": "BETA" },
            ],
            "old_string": "beta",
            "new_string": "BETA",
        }))
        .await
        .unwrap();
    assert_eq!(read(&dir, "a.py"), "ALPHA\n");
    assert_eq!(read(&dir, "b.py"), "BETA\n");
}

#[tokio::test]
async fn a_real_top_level_pair_targets_the_top_level_path() {
    let dir = files(&[("a.py", "alpha gamma\n"), ("b.py", "beta\n")]);
    Edit::new(dir.path())
        .run(json!({
            "path": "a.py",
            "edits": [{ "path": "b.py", "old_string": "beta", "new_string": "BETA" }],
            "old_string": "gamma",
            "new_string": "GAMMA",
        }))
        .await
        .unwrap();
    assert_eq!(read(&dir, "a.py"), "alpha GAMMA\n");
    assert_eq!(read(&dir, "b.py"), "BETA\n");
}

#[tokio::test]
async fn the_top_level_path_may_be_omitted_only_when_every_entry_names_one() {
    let dir = files(&[("a.py", "alpha\n"), ("b.py", "beta\n")]);
    let edit = Edit::new(dir.path());
    edit.run(json!({ "edits": [
        { "path": "a.py", "old_string": "alpha", "new_string": "A" },
        { "path": "b.py", "old_string": "beta", "new_string": "B" },
    ]}))
    .await
    .unwrap();
    assert_eq!(
        (read(&dir, "a.py"), read(&dir, "b.py")),
        ("A\n".into(), "B\n".into())
    );

    let r = edit
        .run(json!({ "edits": [
            { "path": "a.py", "old_string": "A", "new_string": "a" },
            { "old_string": "B", "new_string": "b" },
        ]}))
        .await;
    assert_eq!(message(r), "missing `path`");
    assert_eq!(read(&dir, "a.py"), "A\n");
}

#[test]
fn write_targets_names_every_file_and_a_single_file_call_keeps_its_one_key() {
    let dir = files(&[("a.py", ""), ("b.py", "")]);
    let edit = Edit::new(dir.path());
    let single = json!({ "path": "a.py", "old_string": "x", "new_string": "y" });
    assert_eq!(
        edit.write_targets(&single),
        vec![edit.write_target(&single).unwrap()]
    );
    let multi = json!({ "path": "a.py", "edits": [
        { "old_string": "x", "new_string": "y" },
        { "path": "b.py", "old_string": "x", "new_string": "y" },
    ]});
    let targets = edit.write_targets(&multi);
    assert_eq!(targets.len(), 2);
    assert!(targets[1].ends_with("b.py"), "{targets:?}");
}

/// An in-memory filesystem on "another machine": paths are lexical, the world is remote, and a path
/// can be set to refuse its guarded write (as if it changed after it was read).
#[derive(Default)]
struct MemFs {
    files: Mutex<HashMap<String, (Vec<u8>, u64)>>,
    refuse: Mutex<HashSet<String>>,
}

impl MemFs {
    fn with(files: &[(&str, &str)]) -> Arc<Self> {
        let fs = Self::default();
        for (p, body) in files {
            fs.files
                .lock()
                .unwrap()
                .insert(p.to_string(), (body.as_bytes().to_vec(), 1));
        }
        Arc::new(fs)
    }

    fn get(&self, p: &str) -> String {
        String::from_utf8(self.files.lock().unwrap()[p].0.clone()).unwrap()
    }
}

fn version_time(v: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(v)
}

#[async_trait]
impl FsBackend for MemFs {
    async fn search(&self, _q: &SearchQuery) -> Result<SearchOutcome, FsError> {
        Err(FsError::Backend("unused".into()))
    }
    async fn stat(&self, path: &Path) -> Result<Option<Meta>, FsError> {
        let files = self.files.lock().unwrap();
        Ok(files.get(path.to_str().unwrap()).map(|(b, v)| Meta {
            kind: FileKind::File,
            len: b.len() as u64,
            mtime: Some(version_time(*v)),
            writable: true,
        }))
    }
    async fn read_bytes(&self, path: &Path, offset: u64, max: usize) -> Result<Vec<u8>, FsError> {
        let files = self.files.lock().unwrap();
        let (b, _) = files
            .get(path.to_str().unwrap())
            .ok_or_else(|| FsError::Backend(format!("no such file: {}", path.display())))?;
        let start = (offset as usize).min(b.len());
        Ok(b[start..(start + max).min(b.len())].to_vec())
    }
    async fn write_bytes(&self, path: &Path, bytes: Vec<u8>) -> Result<(), FsError> {
        let mut files = self.files.lock().unwrap();
        let v = files.get(path.to_str().unwrap()).map_or(1, |f| f.1 + 1);
        files.insert(path.to_str().unwrap().to_string(), (bytes, v));
        Ok(())
    }
    async fn write_if_unchanged(
        &self,
        path: &Path,
        bytes: Vec<u8>,
        expected: Option<SystemTime>,
    ) -> Result<bool, FsError> {
        let p = path.to_str().unwrap();
        if self.refuse.lock().unwrap().contains(p) {
            return Ok(false);
        }
        let mut files = self.files.lock().unwrap();
        let entry = files.get_mut(p).expect("planned files exist");
        if Some(version_time(entry.1)) != expected {
            return Ok(false);
        }
        *entry = (bytes, entry.1 + 1);
        Ok(true)
    }
    async fn create_dir_all(&self, _path: &Path) -> Result<(), FsError> {
        Ok(())
    }
    async fn list_dir(
        &self,
        _path: &Path,
        _cap: usize,
        _include_hidden: bool,
    ) -> Result<Vec<DirEntry>, FsError> {
        Err(FsError::Backend("unused".into()))
    }
    async fn glob(&self, _q: &GlobQuery) -> Result<GlobOutcome, FsError> {
        Err(FsError::Backend("unused".into()))
    }
    fn world(&self) -> PathWorld {
        PathWorld::Remote { home: None }
    }
}

#[tokio::test]
async fn a_remote_backend_plans_and_writes_every_file_through_the_backend() {
    // Paths resolve lexically against the sandbox root — nothing about them is asked of this host —
    // and both files are read and written through the backend.
    let fs = MemFs::with(&[
        ("/sbx/repo/a.py", "alpha\n"),
        ("/sbx/repo/pkg/b.py", "beta\n"),
    ]);
    let edit = Edit::new("/sbx/repo").with_backend(fs.clone());
    // The lock keys fold spellings lexically, never via this host's `canonicalize`.
    let spelled = json!({ "path": "a.py", "edits": [
        { "path": "pkg/../pkg/b.py", "old_string": "beta", "new_string": "BETA" },
    ]});
    assert_eq!(
        edit.write_targets(&spelled),
        vec![
            "/sbx/repo/a.py".to_string(),
            "/sbx/repo/pkg/b.py".to_string()
        ]
    );
    edit.run(json!({ "path": "a.py", "edits": [
        { "old_string": "alpha", "new_string": "ALPHA" },
        { "path": "pkg/b.py", "old_string": "beta", "new_string": "BETA" },
    ]}))
    .await
    .unwrap();
    assert_eq!(fs.get("/sbx/repo/a.py"), "ALPHA\n");
    assert_eq!(fs.get("/sbx/repo/pkg/b.py"), "BETA\n");
}

#[tokio::test]
async fn a_file_changed_under_a_multi_file_write_is_reported_with_exactly_what_landed() {
    // Every file planned fine, then b.py moved on before its guarded write: a.py has landed, b.py and
    // the files after it were not written, and the result says which is which.
    let fs = MemFs::with(&[
        ("/sbx/a.py", "alpha\n"),
        ("/sbx/b.py", "beta\n"),
        ("/sbx/c.py", "gamma\n"),
    ]);
    fs.refuse.lock().unwrap().insert("/sbx/b.py".into());
    let r = Edit::new("/sbx")
        .with_backend(fs.clone())
        .run(json!({ "path": "a.py", "edits": [
            { "old_string": "alpha", "new_string": "ALPHA" },
            { "path": "b.py", "old_string": "beta", "new_string": "BETA" },
            { "path": "c.py", "old_string": "gamma", "new_string": "GAMMA" },
        ]}))
        .await;
    assert!(matches!(r, Err(ToolError::Execution(_))), "{r:?}");
    let msg = message(r);
    assert!(
        msg.starts_with("edited /sbx/a.py (1 replacement)\nNOT edited: /sbx/b.py was modified"),
        "{msg}"
    );
    assert!(
        msg.ends_with("NOT edited (not attempted): /sbx/c.py"),
        "{msg}"
    );
    assert_eq!(fs.get("/sbx/a.py"), "ALPHA\n");
    assert_eq!(fs.get("/sbx/b.py"), "beta\n");
    assert_eq!(fs.get("/sbx/c.py"), "gamma\n");
}

#[tokio::test]
async fn a_single_file_stale_write_keeps_its_original_message() {
    let fs = MemFs::with(&[("/sbx/a.py", "alpha\n")]);
    fs.refuse.lock().unwrap().insert("/sbx/a.py".into());
    let r = Edit::new("/sbx")
        .with_backend(fs)
        .run(json!({ "path": "a.py", "old_string": "alpha", "new_string": "A" }))
        .await;
    assert_eq!(
        message(r),
        "/sbx/a.py was modified on disk after it was read; the edit was aborted to avoid \
         discarding that change — re-read the file and retry"
    );
}

#[test]
fn parse_edits_carries_each_entrys_path() {
    let parsed = parse_edits(&json!({ "path": "a.py", "edits": "[{\"path\":\"b.py\",\"old_string\":\"x\",\"new_string\":\"y\"},{\"old_string\":\"p\",\"new_string\":\"q\"}]" }))
        .unwrap();
    let paths: Vec<Option<&str>> = parsed.iter().map(|e| e.path.as_deref()).collect();
    assert_eq!(paths, vec![Some("b.py"), None]);
}

#[tokio::test]
async fn a_file_padded_only_with_no_op_entries_is_skipped_not_fatal() {
    // Luna pads a batch with `old == new` entries. A file that only gets those must not sink the
    // files that really change; the result says it was left alone.
    let dir = files(&[("a.py", "alpha\n"), ("b.py", "beta\n")]);
    let out = Edit::new(dir.path())
        .run(json!({ "path": "", "edits": [
            { "path": "a.py", "old_string": "alpha", "new_string": "ALPHA" },
            { "path": "b.py", "old_string": "beta", "new_string": "beta" },
        ]}))
        .await
        .unwrap();
    assert_eq!(read(&dir, "a.py"), "ALPHA\n");
    assert_eq!(read(&dir, "b.py"), "beta\n");
    let b = dir.path().join("b.py");
    assert!(
        out.text.ends_with(&format!(
            "\nleft {} as it was: its edits replace text with itself",
            b.display()
        )),
        "{}",
        out.text
    );
}

#[tokio::test]
async fn a_multi_file_call_that_changes_nothing_still_errors() {
    let dir = files(&[("a.py", "alpha\n"), ("b.py", "beta\n")]);
    let msg = message(
        Edit::new(dir.path())
            .run(json!({ "path": "a.py", "edits": [
                { "old_string": "alpha", "new_string": "alpha" },
                { "path": "b.py", "old_string": "beta", "new_string": "beta" },
            ]}))
            .await,
    );
    assert!(msg.contains("edit made no changes to"), "{msg}");
}

#[tokio::test]
async fn a_no_op_file_does_not_excuse_a_failing_one() {
    let dir = files(&[("a.py", "alpha\n"), ("b.py", "beta\n"), ("c.py", "gamma\n")]);
    let msg = message(
        Edit::new(dir.path())
            .run(json!({ "path": "a.py", "edits": [
                { "old_string": "alpha", "new_string": "ALPHA" },
                { "path": "b.py", "old_string": "beta", "new_string": "beta" },
                { "path": "c.py", "old_string": "nope", "new_string": "x" },
            ]}))
            .await,
    );
    assert!(msg.contains("not found in edits[2]"), "{msg}");
    assert_eq!(read(&dir, "a.py"), "alpha\n");
}
