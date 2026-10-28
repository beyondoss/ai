//! The local side of a download: extraction of an untrusted archive under a directory, by the rules
//! of FILES-OVER-EXEC.md section 5.8. Nothing is ever written through a symbolic link.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::os::unix::ffi::OsStringExt as _;
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

use super::archive::{Entry, Sink};
use super::{BulkError, Transfer};

fn unsafe_member(name: &[u8], why: &str) -> BulkError {
    BulkError::Archive(format!(
        "unsafe archive member {}: {why}",
        String::from_utf8_lossy(name).escape_debug()
    ))
}

/// A name or link target read the way rule 1 reads it: the components below the root.
fn components(name: &[u8]) -> Result<Vec<OsString>, &'static str> {
    if name.contains(&0) {
        return Err("it has a NUL");
    }
    if name.first() == Some(&b'/') {
        return Err("it is absolute");
    }
    if std::str::from_utf8(name).is_err() {
        return Err("it is not valid UTF-8");
    }
    let mut out = Vec::new();
    for c in name.split(|&b| b == b'/') {
        match c {
            b"" | b"." => {}
            b".." => return Err("it has a \"..\" component"),
            c => out.push(OsString::from_vec(c.to_vec())),
        }
    }
    Ok(out)
}

fn io_ctx(path: &Path, e: std::io::Error) -> BulkError {
    BulkError::Io(std::io::Error::new(
        e.kind(),
        format!("{}: {e}", path.display()),
    ))
}

struct Open {
    file: File,
    path: PathBuf,
    mode: u32,
    mtime: u64,
}

pub(super) struct Extractor {
    root: PathBuf,
    current: Option<Open>,
    /// Directories to finish at the end (mode, mtime), with their depth.
    dirs: Vec<(usize, PathBuf, u32, u64)>,
    /// Regular files this archive wrote, as paths below the root: the targets a hard link may name.
    files: HashSet<PathBuf>,
    pub(super) counts: Transfer,
}

impl Extractor {
    pub(super) fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            current: None,
            dirs: Vec::new(),
            files: HashSet::new(),
            counts: Transfer::default(),
        }
    }

    /// Make every directory above the last of `comps` exist as a real directory.
    fn parents(&self, name: &[u8], comps: &[OsString]) -> Result<PathBuf, BulkError> {
        let mut p = self.root.clone();
        for c in &comps[..comps.len().saturating_sub(1)] {
            p.push(c);
            match fs::symlink_metadata(&p) {
                Ok(m) if m.is_dir() => {}
                Ok(_) => {
                    return Err(unsafe_member(
                        name,
                        "a parent is not a directory (a link or a file)",
                    ));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    fs::DirBuilder::new()
                        .mode(0o755)
                        .create(&p)
                        .map_err(|e| io_ctx(&p, e))?;
                }
                Err(e) => return Err(io_ctx(&p, e)),
            }
        }
        if let Some(last) = comps.last() {
            p.push(last);
        }
        Ok(p)
    }

    /// Remove a non-directory in the way of a member; a directory there is the caller's to handle.
    fn clear(path: &Path) -> Result<bool, BulkError> {
        match fs::symlink_metadata(path) {
            Ok(m) if m.is_dir() => Ok(true),
            Ok(_) => fs::remove_file(path)
                .map(|()| false)
                .map_err(|e| io_ctx(path, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io_ctx(path, e)),
        }
    }

    /// The directories' modes and mtimes, deepest first.
    pub(super) fn finish(&mut self) -> Result<Transfer, BulkError> {
        self.dirs.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, path, mode, mtime) in std::mem::take(&mut self.dirs) {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode & 0o777))
                .map_err(|e| io_ctx(&path, e))?;
            set_mtime(&path, mtime)?;
        }
        Ok(self.counts)
    }
}

fn set_mtime(path: &Path, secs: u64) -> Result<(), BulkError> {
    let f = File::open(path).map_err(|e| io_ctx(path, e))?;
    f.set_modified(UNIX_EPOCH + Duration::from_secs(secs))
        .map_err(|e| io_ctx(path, e))
}

impl Sink for Extractor {
    fn begin(&mut self, e: &Entry) -> Result<(), BulkError> {
        let comps = components(&e.name).map_err(|why| unsafe_member(&e.name, why))?;
        if comps.is_empty() {
            // The root: sets nothing on the directory the caller chose. Its content (a root
            // "file" has none) is skipped by the parser.
            if e.kind == b'5' {
                return Ok(());
            }
            return Err(unsafe_member(&e.name, "it names the root"));
        }
        match e.kind {
            b'5' => {
                let path = self.parents(&e.name, &comps)?;
                if Self::clear(&path)? {
                    // Already a real directory: merge.
                } else {
                    fs::DirBuilder::new()
                        .mode(0o700)
                        .create(&path)
                        .map_err(|e| io_ctx(&path, e))?;
                }
                self.dirs.push((comps.len(), path, e.mode, e.mtime));
                self.counts.dirs += 1;
            }
            b'0' | b'7' => {
                let path = self.parents(&e.name, &comps)?;
                if Self::clear(&path)? {
                    return Err(io_ctx(
                        &path,
                        std::io::Error::other("Is a directory (os error 21)"),
                    ));
                }
                // `create_new` is `O_CREAT|O_EXCL`, which never follows a link at the final component.
                let file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(&path)
                    .map_err(|e| io_ctx(&path, e))?;
                self.files.insert(comps.iter().collect());
                self.counts.files += 1;
                self.current = Some(Open {
                    file,
                    path,
                    mode: e.mode,
                    mtime: e.mtime,
                });
            }
            b'2' => {
                let path = self.parents(&e.name, &comps)?;
                if Self::clear(&path)? {
                    return Err(io_ctx(
                        &path,
                        std::io::Error::other("Is a directory (os error 21)"),
                    ));
                }
                std::os::unix::fs::symlink(OsString::from_vec(e.link.clone()), &path)
                    .map_err(|e| io_ctx(&path, e))?;
                self.counts.links += 1;
            }
            b'1' => {
                let target = components(&e.link).map_err(|why| {
                    unsafe_member(&e.name, &format!("its link target is unsafe: {why}"))
                })?;
                let key: PathBuf = target.iter().collect();
                if target.is_empty() || !self.files.contains(&key) {
                    return Err(unsafe_member(
                        &e.name,
                        "a hard link must name a file this archive already wrote",
                    ));
                }
                let path = self.parents(&e.name, &comps)?;
                if Self::clear(&path)? {
                    return Err(io_ctx(
                        &path,
                        std::io::Error::other("Is a directory (os error 21)"),
                    ));
                }
                let mut src = self.root.clone();
                src.extend(&target);
                // The target was written by this call and `parents` proved its own parents real, but
                // a file this archive replaced by a link since is not a target.
                match fs::symlink_metadata(&src) {
                    Ok(m) if m.is_file() => {}
                    _ => {
                        return Err(unsafe_member(&e.name, "its link target is not a file"));
                    }
                }
                fs::hard_link(&src, &path).map_err(|e| io_ctx(&path, e))?;
                self.files.insert(comps.iter().collect());
                self.counts.links += 1;
            }
            _ => {
                return Err(BulkError::Io(std::io::Error::other(format!(
                    "{}: unsupported member type {:?}",
                    String::from_utf8_lossy(&e.name),
                    e.kind as char
                ))));
            }
        }
        Ok(())
    }

    fn data(&mut self, bytes: &[u8]) -> Result<(), BulkError> {
        if let Some(cur) = &mut self.current {
            cur.file
                .write_all(bytes)
                .map_err(|e| io_ctx(&cur.path, e))?;
            self.counts.bytes += bytes.len() as u64;
        }
        Ok(())
    }

    fn end(&mut self) -> Result<(), BulkError> {
        if let Some(cur) = self.current.take() {
            cur.file
                .set_permissions(fs::Permissions::from_mode(cur.mode & 0o777))
                .map_err(|e| io_ctx(&cur.path, e))?;
            cur.file
                .set_modified(UNIX_EPOCH + Duration::from_secs(cur.mtime))
                .map_err(|e| io_ctx(&cur.path, e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::archive::{Entry, Parser, pad_to_block, write_end, write_header};
    use super::*;

    fn member(name: &str, kind: u8, data: &[u8], link: &str, mode: u32) -> Vec<u8> {
        let mut out = Vec::new();
        write_header(
            &mut out,
            &Entry {
                name: name.into(),
                kind,
                mode,
                size: data.len() as u64,
                mtime: 1_500_000_000,
                link: link.into(),
            },
        );
        out.extend_from_slice(data);
        pad_to_block(&mut out, data.len() as u64);
        out
    }

    fn run(root: &Path, members: &[Vec<u8>]) -> Result<Transfer, BulkError> {
        std::fs::create_dir_all(root).unwrap();
        let mut a: Vec<u8> = members.concat();
        write_end(&mut a);
        let mut p = Parser::new();
        let mut ex = Extractor::new(root);
        p.feed(&a, &mut ex)?;
        p.finish()?;
        ex.finish()
    }

    fn refused(r: Result<Transfer, BulkError>, why: &str) {
        match r {
            Err(BulkError::Archive(m)) => {
                assert!(m.starts_with("unsafe archive member"), "{m}");
                assert!(m.contains(why), "{m} should say {why}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_name_that_leaves_the_target_is_refused_and_nothing_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        for name in ["../x", "a/../../x", "/tmp/abs-escape", "a/.."] {
            let r = run(&root, &[member(name, b'0', b"x", "", 0o644)]);
            assert!(matches!(r, Err(BulkError::Archive(_))), "{name:?}: {r:?}");
        }
        assert!(!dir.path().join("x").exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn nothing_is_written_through_a_link() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        // A link the archive made, and one that was already there.
        refused(
            run(
                &root,
                &[
                    member("l", b'2', b"", "../outside", 0o777),
                    member("l/f", b'0', b"pwned", "", 0o644),
                ],
            ),
            "a parent is not a directory",
        );
        std::os::unix::fs::symlink(&outside, root.join("pre")).unwrap();
        refused(
            run(&root, &[member("pre/f", b'0', b"pwned", "", 0o644)]),
            "a parent is not a directory",
        );
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
        // A file or directory member replaces the link; the link's target is untouched.
        run(&root, &[member("pre", b'0', b"now a file", "", 0o600)]).unwrap();
        assert_eq!(std::fs::read(root.join("pre")).unwrap(), b"now a file");
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
    }

    #[test]
    fn a_hard_link_names_only_a_file_the_archive_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        refused(
            run(&root, &[member("h", b'1', b"", "/etc/passwd", 0o644)]),
            "link target",
        );
        refused(
            run(&root, &[member("h", b'1', b"", "never", 0o644)]),
            "already wrote",
        );
        let t = run(
            &root,
            &[
                member("a", b'0', b"data", "", 0o644),
                member("b", b'1', b"", "./a", 0o644),
            ],
        )
        .unwrap();
        assert_eq!((t.files, t.links), (1, 1));
        assert_eq!(std::fs::read(root.join("b")).unwrap(), b"data");
    }

    #[test]
    fn directory_modes_are_applied_after_their_content() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        run(
            &root,
            &[
                member("d/", b'5', b"", "", 0o555),
                member("d/f", b'0', b"f", "", 0o4755),
            ],
        )
        .unwrap();
        let mode = |p: &str| {
            std::fs::metadata(root.join(p))
                .unwrap()
                .permissions()
                .mode()
                & 0o7777
        };
        assert_eq!(mode("d"), 0o555);
        assert_eq!(mode("d/f"), 0o755, "set-id bits are not carried");
        // So the temp directory can be removed.
        std::fs::set_permissions(root.join("d"), std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn an_unsupported_member_type_is_io_and_a_file_over_a_directory_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        assert!(matches!(
            run(&root, &[member("dev", b'3', b"", "", 0o644)]),
            Err(BulkError::Io(e)) if e.to_string().contains("unsupported member type")
        ));
        std::fs::create_dir_all(root.join("d")).unwrap();
        assert!(matches!(
            run(&root, &[member("d", b'0', b"x", "", 0o644)]),
            Err(BulkError::Io(e)) if e.to_string().contains("Is a directory")
        ));
    }
}
