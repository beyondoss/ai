//! The local trees of the bulk-copy ops (`sdk/conformance/DRIVER.md`, "Trees"): building one from a
//! request, listing one, and the sentinel layout that makes a write outside the extraction target
//! fail the op. Shared by the Rust and MCP drivers, which include this file by path as well.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

use std::io::{Read as _, Write as _};
use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

use base64::Engine as _;
use serde_json::{Value, json};

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn err(kind: &str, detail: impl Into<String>) -> Value {
    json!({ "kind": kind, "detail": detail.into() })
}

fn io(e: std::io::Error) -> Value {
    err("io", e.to_string())
}

fn mode_str(m: u32) -> String {
    format!("{:04o}", m & 0o7777)
}

/// Build the tree `entries` describes under `root`. Directory modes are applied last, deepest first.
pub fn materialize(root: &Path, entries: &[Value], outside: &Path) -> Result<(), Value> {
    let mut dirs: Vec<(PathBuf, u32)> = Vec::new();
    let mut times: Vec<(PathBuf, u64)> = Vec::new();
    for e in entries {
        let rel = e["path"]
            .as_str()
            .ok_or_else(|| err("invalid_request", "an entry needs a `path`"))?;
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            // The directories this creates are 0755 whatever the umask.
            let mut missing = Vec::new();
            let mut p = parent;
            while p != root && !p.exists() {
                missing.push(p.to_path_buf());
                p = p.parent().unwrap_or(root);
            }
            std::fs::create_dir_all(parent).map_err(io)?;
            dirs.extend(missing.into_iter().map(|m| (m, 0o755)));
        }
        let mode = match e["mode"].as_str() {
            Some(m) => u32::from_str_radix(m, 8)
                .map_err(|_| err("invalid_request", format!("bad mode {m:?}")))?,
            None => 0o644,
        };
        match e["type"].as_str().unwrap_or("file") {
            "dir" => {
                std::fs::create_dir_all(&path).map_err(io)?;
                dirs.retain(|(p, _)| *p != path);
                dirs.push((path.clone(), e["mode"].as_str().map_or(0o755, |_| mode)));
            }
            "symlink" => {
                let target = e["target"]
                    .as_str()
                    .ok_or_else(|| err("invalid_request", "a symlink needs a `target`"))?
                    .replace("${OUTSIDE}", &outside.to_string_lossy());
                std::os::unix::fs::symlink(target, &path).map_err(io)?;
            }
            "fifo" => {
                let ok = std::process::Command::new("mkfifo")
                    .arg(&path)
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                if !ok {
                    return Err(err("io", "mkfifo failed"));
                }
            }
            _ => {
                let mut f = std::fs::File::create(&path).map_err(io)?;
                if let Some(p) = e["pattern_b64"].as_str() {
                    let pat = B64
                        .decode(p)
                        .map_err(|_| err("invalid_request", "`pattern_b64` is not base64"))?;
                    let mut left = e["size"].as_u64().unwrap_or(0);
                    let block: Vec<u8> = pat.iter().cycle().take(1 << 20).copied().collect();
                    // Every write is a whole number of patterns, so the file is the pattern repeated.
                    let step = block.len() - block.len() % pat.len().max(1);
                    while left > 0 {
                        let n = left.min(step as u64) as usize;
                        f.write_all(&block[..n]).map_err(io)?;
                        left -= n as u64;
                    }
                } else if let Some(d) = e["data_b64"].as_str() {
                    let data = B64
                        .decode(d)
                        .map_err(|_| err("invalid_request", "`data_b64` is not base64"))?;
                    f.write_all(&data).map_err(io)?;
                }
                f.set_permissions(std::fs::Permissions::from_mode(mode))
                    .map_err(io)?;
                if let Some(t) = e["mtime"].as_u64() {
                    times.push((path.clone(), t));
                }
            }
        }
    }
    for (p, t) in times {
        std::fs::File::open(&p)
            .and_then(|f| f.set_modified(UNIX_EPOCH + Duration::from_secs(t)))
            .map_err(io)?;
    }
    dirs.sort_by_key(|(p, _)| std::cmp::Reverse(p.components().count()));
    for (p, m) in dirs {
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(m)).map_err(io)?;
    }
    Ok(())
}

/// What is under `root`, sorted by path: `{path, type, mode, size, sha256, target, mtime}`.
pub fn describe(root: &Path) -> Result<Vec<Value>, Value> {
    use sha2::Digest as _;
    fn walk(root: &Path, dir: &Path, out: &mut Vec<Value>) -> Result<(), Value> {
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .map_err(io)?
            .map(|r| r.map(|d| d.path()))
            .collect::<Result<_, _>>()
            .map_err(io)?;
        names.sort();
        for p in names {
            let m = std::fs::symlink_metadata(&p).map_err(io)?;
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
            let ft = m.file_type();
            if ft.is_symlink() {
                let t = std::fs::read_link(&p).map_err(io)?;
                out.push(json!({"path": rel, "type": "symlink", "target": t.to_string_lossy()}));
            } else if ft.is_dir() {
                out.push(
                    json!({"path": rel, "type": "dir", "mode": mode_str(m.permissions().mode()),
                    "mtime": m.mtime()}),
                );
                // A directory nobody may enter is listed, not entered.
                if m.permissions().mode() & 0o500 == 0o500 {
                    walk(root, &p, out)?;
                }
            } else {
                let mut f = std::fs::File::open(&p).map_err(io)?;
                let mut h = sha2::Sha256::new();
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let n = f.read(&mut buf).map_err(io)?;
                    if n == 0 {
                        break;
                    }
                    h.update(&buf[..n]);
                }
                out.push(
                    json!({"path": rel, "type": "file", "mode": mode_str(m.permissions().mode()),
                    "size": m.len(), "sha256": hex::encode(h.finalize()), "mtime": m.mtime()}),
                );
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    Ok(out)
}

fn temp(base: Option<&Path>) -> Result<tempfile::TempDir, Value> {
    match base {
        Some(b) => tempfile::tempdir_in(b),
        None => tempfile::tempdir(),
    }
    .map_err(io)
}

/// A request's `entries`, built in a fresh directory: the source of an upload.
pub struct Tree {
    tmp: tempfile::TempDir,
}

impl Tree {
    pub fn build(req: &Value) -> Result<Self, Value> {
        Self::build_in(None, req)
    }

    /// [`Self::build`] below `base` (a driver whose client may only read below one directory).
    pub fn build_in(base: Option<&Path>, req: &Value) -> Result<Self, Value> {
        let entries = req["entries"]
            .as_array()
            .ok_or_else(|| err("invalid_request", "`entries` must be a list"))?;
        let tmp = temp(base)?;
        std::fs::create_dir_all(tmp.path().join("src")).map_err(io)?;
        materialize(&tmp.path().join("src"), entries, tmp.path())?;
        Ok(Self { tmp })
    }

    /// The directory the entries were built in.
    pub fn dir(&self) -> PathBuf {
        self.tmp.path().join("src")
    }
}

/// The layout of a download-side op: the target `local` (first holding `prefill`) and `outside`,
/// an empty directory a `${OUTSIDE}` link points at, in one temporary directory. `finish` fails the
/// op with `io` if anything else appeared in that directory or `outside` is not empty: a write that
/// escaped its target, whatever the client answered.
pub struct Layout {
    tmp: tempfile::TempDir,
    pub local: PathBuf,
    outside: PathBuf,
}

impl Layout {
    pub fn new(prefill: &Value) -> Result<Self, Value> {
        Self::new_in(None, prefill)
    }

    /// [`Self::new`] below `base`.
    pub fn new_in(base: Option<&Path>, prefill: &Value) -> Result<Self, Value> {
        let tmp = temp(base)?;
        let local = tmp.path().join("local");
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&local).map_err(io)?;
        std::fs::create_dir_all(&outside).map_err(io)?;
        if let Some(pre) = prefill.as_array() {
            materialize(&local, pre, &outside)?;
        }
        Ok(Self {
            tmp,
            local,
            outside,
        })
    }

    /// Settle the op: `counts` is the result so far (`{files, dirs, links, bytes}`) or the client's
    /// error; the listing of `local` is added to a result.
    pub fn finish(self, counts: Result<Value, Value>) -> Result<Value, Value> {
        for entry in std::fs::read_dir(self.tmp.path()).map_err(io)? {
            let name = entry.map_err(io)?.file_name();
            if name != "local" && name != "outside" {
                return Err(err(
                    "io",
                    format!(
                        "the extraction wrote outside its directory: {}",
                        name.to_string_lossy()
                    ),
                ));
            }
        }
        if std::fs::read_dir(&self.outside)
            .map_err(io)?
            .next()
            .is_some()
        {
            return Err(err(
                "io",
                "the extraction wrote through a link into `outside`",
            ));
        }
        let mut v = counts?;
        v["entries"] = Value::Array(describe(&self.local)?);
        Ok(v)
    }
}
