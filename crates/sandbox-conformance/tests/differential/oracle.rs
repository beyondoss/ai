//! Ground truth for the local backends: the real filesystem, read with `std::fs`, never through a
//! client. The oracle files are the `ShellFs` property test's (`truth.rs`: snapshots and the plans
//! that say when a write or mkdir must succeed or fail; `search_truth.rs`: a brute-force search),
//! included, not copied.
//!
//! [`before`] looks at the tree and the request, [`after`] checks the driver's answer and the tree it
//! left. A defect the property test already knows (N1..N4) is returned as `Ok(Some(id))`: tolerated,
//! counted, and the operation is left out of the cross-driver comparison, exactly as the property
//! test's `known(..)` does. When a fix lands its entry is deleted from [`after`] and the divergence
//! becomes an ordinary failure.
use std::path::{Path, PathBuf};

use base64::Engine as _;
use serde_json::Value;

use super::ops::Op;

#[path = "../../../agent/tests/shellfs_props_env/search_truth.rs"]
pub mod search_truth;
#[path = "../../../agent/tests/shellfs_props_env/truth.rs"]
pub mod truth;

use search_truth::{Engine, Spec, Th};
use truth::{Kind, Node, Plan, Snap, secs, snapshot};

pub struct Ctx {
    pub root: PathBuf,
    pub engine: Engine,
    pub find_printf: bool,
    /// `busybox-grep` is not an engine the brute-force search models (it applies the glob to every
    /// path, explicit files included), so its answers are checked for soundness only.
    pub busybox: bool,
}

pub struct Pre {
    snap: Snap,
    plan: Option<Plan>,
    /// The mtime (whole seconds, following links) of the path a guarded write addresses, before it.
    mtime: Option<u64>,
    /// Where a write to the path lands (links followed as the write script does), before it: a
    /// write replaces a symlink loop, after which the path resolves to itself.
    target: Option<Option<PathBuf>>,
}

fn path(ctx: &Ctx, rel: &str) -> PathBuf {
    if rel.is_empty() {
        ctx.root.clone()
    } else {
        ctx.root.join(rel)
    }
}

pub fn before(op: &Op, ctx: &Ctx) -> Pre {
    let snap = snapshot(&ctx.root);
    let plan = match op {
        Op::Write { rel, .. } | Op::WriteIf { rel, .. } => Some(truth::plan_write(&path(ctx, rel))),
        Op::Mkdir { rel } => Some(truth::plan_mkdir(&path(ctx, rel))),
        _ => None,
    };
    let mtime = match op {
        Op::WriteIf { rel, .. } => std::fs::metadata(path(ctx, rel))
            .ok()
            .and_then(|m| m.modified().ok())
            .map(secs),
        _ => None,
    };
    let target = match op {
        Op::Write { rel, .. } | Op::WriteIf { rel, .. } => {
            Some(truth::resolve_leaf(&path(ctx, rel)))
        }
        _ => None,
    };
    Pre {
        snap,
        plan,
        mtime,
        target,
    }
}

fn kind_of(e: &Value) -> Kind {
    match e["kind"].as_str().unwrap_or("") {
        "not_found" => Kind::NotFound,
        "is_directory" => Kind::IsDirectory,
        "not_a_directory" => Kind::NotADirectory,
        "permission_denied" => Kind::PermissionDenied,
        "io" => Kind::Io,
        "too_large" => Kind::TooLarge,
        "invalid_query" => Kind::InvalidQuery,
        "timeout" => Kind::Timeout,
        _ => Kind::Transport,
    }
}

/// A path with a newline in it splits the shell's one-line error, so the kind falls back to `io`
/// (FILES-OVER-EXEC 3.3 reads the first line only). Allowed.
fn kind_ok(kinds: &[Kind], k: Kind, p: &Path) -> bool {
    kinds.contains(&k) || (k == Kind::Io && (p.to_string_lossy().contains('\n') || has_loop(p)))
}

/// A symlink loop on the way ("Too many levels of symbolic links") is none of the errors
/// FILES-OVER-EXEC 3.3 names, so it is `io`.
fn has_loop(p: &Path) -> bool {
    p.ancestors()
        .any(|a| std::fs::metadata(a).is_err_and(|e| e.raw_os_error() == Some(40)))
}

fn node_desc(n: &Node) -> String {
    match n {
        Node::File { mode, bytes, .. } => format!(
            "file mode={mode:o} len={} head={:?}",
            bytes.len(),
            &bytes[..bytes.len().min(24)]
        ),
        Node::Dir { mode, kids } => format!("dir mode={mode:o} kids={kids:?}"),
        Node::Link(t) => format!("symlink -> {t:?}"),
        Node::Other { mode } => format!("other mode={mode:o}"),
    }
}

fn diff(actual: &Snap, expected: &Snap, root: &Path) -> String {
    let rel = |p: &Path| p.strip_prefix(root).unwrap_or(p).display().to_string();
    let mut out = vec![];
    for (p, a) in actual {
        match expected.get(p) {
            None => out.push(format!("unexpected {:?}: {}", rel(p), node_desc(a))),
            Some(e) if e != a => out.push(format!(
                "{:?}: actual {} / expected {}",
                rel(p),
                node_desc(a),
                node_desc(e)
            )),
            _ => {}
        }
    }
    for (p, e) in expected {
        if !actual.contains_key(p) {
            out.push(format!("missing {:?}: {}", rel(p), node_desc(e)));
        }
    }
    out.join("\n")
}

fn unchanged(ctx: &Ctx, pre: &Pre, what: &str) -> Result<(), String> {
    let post = snapshot(&ctx.root);
    if post == pre.snap {
        Ok(())
    } else {
        Err(format!(
            "{what} changed the tree:\n{}",
            diff(&post, &pre.snap, &ctx.root)
        ))
    }
}

fn data_of(ok: &Value, field: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(ok[field].as_str().unwrap_or(""))
        .unwrap_or_default()
}

/// What the tree is if the write was applied: only the file the path resolves to changes.
fn applied(pre: &Snap, post: &Snap, target: &Path, data: &[u8]) -> Result<Snap, String> {
    let mut e = pre.clone();
    let actual = post.get(target);
    match (e.get_mut(target), actual) {
        (Some(Node::File { bytes, mtime, .. }), Some(Node::File { mtime: am, .. })) => {
            *bytes = data.to_vec();
            *mtime = *am;
        }
        (Some(Node::File { .. }), _) => {} // the comparison below reports it
        (Some(_), _) => {}
        (None, Some(Node::File { mode, mtime, .. })) => {
            // A new file's mode is the umask's business, but it must be an ordinary one.
            if mode & 0o7000 != 0 || mode & 0o600 != 0o600 {
                return Err(format!("the new file has the odd mode {mode:o}"));
            }
            e.insert(
                target.to_path_buf(),
                Node::File {
                    mode: *mode,
                    mtime: *mtime,
                    bytes: data.to_vec(),
                },
            );
            if let (Some(par), Some(name)) = (target.parent(), target.file_name())
                && let Some(Node::Dir { kids, .. }) = e.get_mut(par)
            {
                kids.insert(name.to_os_string());
            }
        }
        (None, _) => {}
    }
    Ok(e)
}

pub fn after(
    op: &Op,
    req: &Value,
    resp: &Value,
    pre: &Pre,
    ctx: &Ctx,
) -> Result<Option<&'static str>, String> {
    let err = resp.get("error");
    let ok = resp.get("ok");
    match op {
        Op::Write { rel, .. } | Op::WriteIf { rel, .. } => {
            let p = path(ctx, rel);
            let data = match op {
                Op::Write { data, .. } | Op::WriteIf { data, .. } => data,
                _ => unreachable!(),
            };
            let post = snapshot(&ctx.root);
            let temps = truth::temps(&post);
            if temps.iter().any(|t| !pre.snap.contains_key(t)) {
                return Err(format!("{} left a temp file behind: {temps:?}", op.name()));
            }
            if let Op::WriteIf { .. } = op {
                let expected = req["expected_mtime"].as_u64();
                let cur = pre.mtime;
                if cur != expected {
                    // The guard fails first (FILES-OVER-EXEC 5.4), whatever the write would do.
                    return match err {
                        Some(e) if kind_str(e) == "conflict" && post == pre.snap => Ok(None),
                        Some(e) if kind_str(e) == "conflict" => Err(format!(
                            "a conflicting write changed the tree:\n{}",
                            diff(&post, &pre.snap, &ctx.root)
                        )),
                        _ => Err(format!(
                            "write_if_unchanged expected mtime {expected:?} but the file's is {cur:?}: a lost update, answered {resp}"
                        )),
                    };
                }
            }
            let Some(target) = pre.target.clone().flatten() else {
                return Ok(None); // a symlink loop: unspecified
            };
            let plan = pre.plan.clone().expect("a write has a plan");
            match err {
                None => {
                    if let Plan::Fail(k) = &plan {
                        return Err(format!(
                            "{} reported success but the spec requires an error {k:?}",
                            op.name()
                        ));
                    }
                    let want = applied(&pre.snap, &post, &target, data)?;
                    if post != want {
                        return Err(format!(
                            "the tree is not 'only the target changed to the new bytes':\n{}",
                            diff(&post, &want, &ctx.root)
                        ));
                    }
                    Ok(None)
                }
                Some(e) => {
                    let k = kind_of(e);
                    if k == Kind::Transport || k == Kind::Timeout {
                        return Err(format!("{} failed as {}: {resp}", op.name(), kind_str(e)));
                    }
                    if post != pre.snap {
                        return Err(format!(
                            "a failed {} changed the tree:\n{}",
                            op.name(),
                            diff(&post, &pre.snap, &ctx.root)
                        ));
                    }
                    match &plan {
                        Plan::Succeed
                            if k == Kind::PermissionDenied
                                && std::fs::metadata(&target).is_ok_and(|m| m.is_file())
                                && !truth::can(&target, 4) =>
                        {
                            Ok(Some("N4"))
                        }
                        Plan::Succeed => Err(format!(
                            "{} {rel:?} must succeed but failed ({}): {resp}",
                            op.name(),
                            kind_str(e)
                        )),
                        Plan::Fail(kinds)
                            if !kind_ok(kinds, k, &p) && !kind_ok(kinds, k, &target) =>
                        {
                            Err(format!(
                                "{} {rel:?} failed as {} but the spec allows {kinds:?}",
                                op.name(),
                                kind_str(e)
                            ))
                        }
                        _ => Ok(None),
                    }
                }
            }
        }
        Op::Mkdir { rel } => {
            let p = path(ctx, rel);
            let post = snapshot(&ctx.root);
            let plan = pre.plan.clone().expect("mkdir has a plan");
            match err {
                None => {
                    if let Plan::Fail(k) = &plan {
                        return Err(format!(
                            "mkdir reported success but the spec requires {k:?}"
                        ));
                    }
                    if !p.is_dir() {
                        return Err(format!("mkdir said ok but {rel:?} is not a directory"));
                    }
                    for (q, n) in &pre.snap {
                        if let Node::File { .. } | Node::Link(_) | Node::Other { .. } = n
                            && post.get(q) != Some(n)
                        {
                            return Err(format!("mkdir changed {q:?}"));
                        }
                    }
                    Ok(None)
                }
                Some(e) => {
                    let k = kind_of(e);
                    if post != pre.snap {
                        return Err(format!(
                            "a failed mkdir changed the tree:\n{}",
                            diff(&post, &pre.snap, &ctx.root)
                        ));
                    }
                    match &plan {
                        Plan::Succeed => {
                            Err(format!("mkdir {rel:?} must succeed but failed: {resp}"))
                        }
                        Plan::Fail(kinds) if !kind_ok(kinds, k, &p) => Err(format!(
                            "mkdir {rel:?} failed as {} but the spec allows {kinds:?}",
                            kind_str(e)
                        )),
                        _ => Ok(None),
                    }
                }
            }
        }
        Op::Stat { rel } => {
            let p = path(ctx, rel);
            unchanged(ctx, pre, "stat")?;
            let Some(ok) = ok else {
                return Err(format!("stat failed: {resp}"));
            };
            let want = std::fs::metadata(&p).ok();
            match (ok["exists"].as_bool(), want) {
                (Some(false), None) => Ok(None),
                (Some(false), Some(m)) => Err(format!(
                    "stat says {rel:?} is absent but std::fs finds it: {m:?}"
                )),
                (Some(true), None) => Err(format!(
                    "stat says {rel:?} exists but std::fs does not find it"
                )),
                (Some(true), Some(m)) => {
                    let kind = if m.is_dir() {
                        "dir"
                    } else if m.is_file() {
                        "file"
                    } else {
                        "other"
                    };
                    if ok["kind"] != kind {
                        return Err(format!("stat kind {} != {kind}", ok["kind"]));
                    }
                    if ok["size"].as_u64() != Some(m.len()) {
                        return Err(format!("stat size {} != {}", ok["size"], m.len()));
                    }
                    let ts = m.modified().map(secs).unwrap();
                    if ok["mtime"].as_u64() != Some(ts) {
                        return Err(format!("stat mtime {} != {ts}", ok["mtime"]));
                    }
                    if ok["writable"].as_bool() != Some(truth::can(&p, 2)) {
                        return Err(format!(
                            "stat writable {} != {}",
                            ok["writable"],
                            truth::can(&p, 2)
                        ));
                    }
                    Ok(None)
                }
                _ => Err(format!("stat answered {resp}")),
            }
        }
        Op::Read { rel, offset, max } => {
            let p = path(ctx, rel);
            unchanged(ctx, pre, "read")?;
            let want = read_expect(&p, *offset, max.unwrap_or(u64::MAX));
            match (ok, err, want) {
                (Some(ok), _, Ok(w)) => {
                    let got = data_of(ok, "data_b64");
                    if got != w {
                        return Err(format!(
                            "read bytes differ: got {} bytes, std::fs range has {} (first difference at {:?})",
                            got.len(),
                            w.len(),
                            got.iter().zip(&w).position(|(x, y)| x != y)
                        ));
                    }
                    Ok(None)
                }
                (None, Some(_), Ok(w)) => Err(format!(
                    "read {rel:?} must return {} bytes but failed: {resp}",
                    w.len()
                )),
                (Some(_), _, Err(k)) => Err(format!(
                    "read of {rel:?} succeeded but the spec requires an error {k:?}"
                )),
                (None, Some(e), Err(kinds)) => {
                    if kind_ok(&kinds, kind_of(e), &p) {
                        Ok(None)
                    } else {
                        Err(format!(
                            "read {rel:?} failed as {} but the spec allows {kinds:?}",
                            kind_str(e)
                        ))
                    }
                }
                _ => Err(format!("read answered {resp}")),
            }
        }
        Op::List { rel, hidden, limit } => list(ctx, pre, rel, *hidden, *limit, resp),
        Op::Search {
            rel,
            pattern,
            literal,
            ignore_case,
            glob,
            limit,
            before,
            after,
            raw,
        } => {
            unchanged(ctx, pre, "search")?;
            if *raw {
                return Ok(None);
            }
            // FILES-OVER-EXEC 5.7: busybox grep "cannot represent" a path with a newline in it, and
            // it mis-splits any `-DIGITS-`; the rung is the last resort and is compared only with
            // the other drivers when the tree has such a name.
            if ctx.busybox && pre.snap.keys().any(|p| p.to_string_lossy().contains('\n')) {
                return Ok(None);
            }
            let root = path(ctx, rel);
            let spec = Spec {
                pattern: pattern.clone(),
                literal: *literal,
                ignore_case: *ignore_case,
                glob: glob.clone(),
                limit: *limit,
                before: *before,
                after: *after,
            };
            let map = |p: &Path| p.to_string_lossy().into_owned();
            let mut want = search_truth::expect(&root, &map, &spec, ctx.engine);
            want.exact &= !ctx.busybox;
            // GNU grep applies `--include`/`--exclude` to a file named on the command line, ripgrep
            // searches it whatever the glob says: the model is ripgrep's.
            want.exact &= !(ctx.engine != Engine::Rg && glob.is_some() && root.is_file());
            match (ok, err) {
                (None, Some(e)) => {
                    let k = kind_of(e);
                    // The busybox rung is forced on a GNU grep here (the container backends have the
                    // real busybox), whose "binary file matches" notice is on stderr and so reads as
                    // a rejected pattern; busybox prints it on stdout.
                    if ctx.busybox
                        && e["detail"]
                            .as_str()
                            .is_some_and(|d| d.contains("binary file matches"))
                    {
                        return Ok(None);
                    }
                    let newline = root.to_string_lossy().contains('\n')
                        || pre.snap.keys().any(|p| p.to_string_lossy().contains('\n'));
                    if k == Kind::InvalidQuery && (want.root_missing || want.unreadable) && newline
                    {
                        return Ok(Some("N2"));
                    }
                    if want.root_below_file && k == Kind::InvalidQuery {
                        return Ok(Some("N3"));
                    }
                    if want.root_missing {
                        if k == Kind::NotFound || (want.root_below_file && k == Kind::NotADirectory)
                        {
                            return Ok(None);
                        }
                        return Err(format!(
                            "search of the missing root {rel:?} must fail as not_found, got {}",
                            kind_str(e)
                        ));
                    }
                    if want.unreadable {
                        return Err(format!(
                            "search lost the matches it could read because a path was unreadable: {resp}"
                        ));
                    }
                    Err(format!("search {pattern:?} failed: {resp}"))
                }
                (Some(ok), None) => {
                    if want.root_missing {
                        return Err(format!(
                            "search of the missing root {rel:?} succeeded: {resp}"
                        ));
                    }
                    let hits: Vec<Th> = ok["hits"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|h| Th {
                            path: h["path"].as_str().unwrap_or("").to_string(),
                            line: h["line"].as_u64().unwrap_or(0) as usize,
                            text: h["text"].as_str().unwrap_or("").to_string(),
                            is_match: h["is_match"].as_bool().unwrap_or(false),
                        })
                        .collect();
                    let truncated = ok["truncated"].as_bool().unwrap_or(false);
                    if want.unreadable && ok["note"].is_null() {
                        return Err("search over an unreadable path reported no note".into());
                    }
                    if hits.iter().filter(|h| h.is_match).count() > *limit {
                        return Err(format!("search returned more than limit={limit} matches"));
                    }
                    if want.exact {
                        if hits != want.hits {
                            let first = hits
                                .iter()
                                .zip(&want.hits)
                                .position(|(a, b)| a != b)
                                .unwrap_or(hits.len().min(want.hits.len()));
                            return Err(format!(
                                "search differs from the brute-force search at hit {first}:\n got  ({}) {:?}\n want ({}) {:?}",
                                hits.len(),
                                hits.get(first),
                                want.hits.len(),
                                want.hits.get(first)
                            ));
                        }
                        if truncated != want.truncated {
                            return Err(format!(
                                "search truncated={truncated} but expected {}",
                                want.truncated
                            ));
                        }
                    } else if let Some(h) = hits.iter().find(|h| !want.hits.contains(h)) {
                        return Err(format!(
                            "search returned a hit that is not in the tree: {h:?}"
                        ));
                    }
                    Ok(None)
                }
                _ => Err(format!("search answered {resp}")),
            }
        }
        Op::Exec {
            command,
            stdin,
            stream,
            timeout_ms,
            args,
            ..
        } => {
            // The only exec with an answer known in advance: `cat` echoes its stdin.
            if command == "cat" && !stream && timeout_ms.is_none() && args.is_empty() {
                let got = ok.map(|o| data_of(o, "stdout_b64"));
                if got.as_deref() != stdin.as_deref() {
                    return Err(format!("cat did not echo its stdin: {resp}"));
                }
            }
            Ok(None)
        }
    }
}

fn kind_str(e: &Value) -> &str {
    e["kind"].as_str().unwrap_or("?")
}

/// FILES-OVER-EXEC 5.2 against the file itself.
fn read_expect(p: &Path, offset: u64, max: u64) -> Result<Vec<u8>, Vec<Kind>> {
    match std::fs::metadata(p) {
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(vec![Kind::NotFound, Kind::PermissionDenied]);
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(vec![Kind::NotFound]),
        Err(_) => return Err(vec![Kind::NotFound, Kind::NotADirectory]),
        Ok(m) if m.is_dir() => return Err(vec![Kind::IsDirectory]),
        Ok(_) => {}
    }
    if !truth::can(p, 4) {
        return Err(vec![Kind::PermissionDenied]);
    }
    let data = std::fs::read(p).unwrap_or_default();
    let lo = offset.min(data.len() as u64) as usize;
    let hi = (lo as u64).saturating_add(max).min(data.len() as u64) as usize;
    Ok(data[lo..hi].to_vec())
}

fn list(
    ctx: &Ctx,
    pre: &Pre,
    rel: &str,
    hidden: bool,
    limit: Option<usize>,
    resp: &Value,
) -> Result<Option<&'static str>, String> {
    let p = path(ctx, rel);
    unchanged(ctx, pre, "list")?;
    let cap = limit.unwrap_or(10_000);
    let meta = std::fs::metadata(&p);
    let mut want: Vec<(String, u8, u64)> = vec![];
    let mut readable = true;
    if let Ok(m) = &meta
        && m.is_dir()
    {
        if truth::can(&p, 5) {
            for e in std::fs::read_dir(&p).unwrap().flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if !hidden && name.starts_with('.') {
                    continue;
                }
                let followed = std::fs::metadata(e.path());
                if let Err(err) = &followed
                    && err.raw_os_error() == Some(40)
                    && ctx.find_printf
                {
                    // A symlink loop: `%Y` is `L`, which FILES-OVER-EXEC 5.6 maps to `other`.
                    let own = std::fs::symlink_metadata(e.path()).unwrap();
                    want.push((name, 2, own.len()));
                    continue;
                }
                let Ok(follow) = followed else {
                    // `find -print0` (the busybox rung) lists a broken link; `%Y` (the other)
                    // reports it as `N` and the client skips it.
                    if !ctx.find_printf {
                        want.push((name, 0, 0));
                    }
                    continue;
                };
                let own = std::fs::symlink_metadata(e.path()).unwrap();
                let k = if follow.is_dir() {
                    1
                } else if follow.is_file() {
                    0
                } else {
                    2
                };
                want.push((name, k, own.len()));
            }
        } else {
            readable = false;
        }
    }
    match (resp.get("ok"), resp.get("error")) {
        (None, Some(e)) => {
            let k = kind_of(e);
            let ok = match &meta {
                Err(_) => {
                    matches!(
                        k,
                        Kind::NotFound | Kind::PermissionDenied | Kind::NotADirectory
                    ) || std::fs::symlink_metadata(&p).is_ok()
                }
                Ok(_) => !readable && matches!(k, Kind::PermissionDenied | Kind::Io),
            };
            if ok {
                Ok(None)
            } else {
                Err(format!("list {rel:?} failed as {}: {resp}", kind_str(e)))
            }
        }
        (Some(ok), None) => {
            let entries = ok["entries"].as_array().cloned().unwrap_or_default();
            let dangling = meta.is_err()
                && std::fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink());
            if meta.is_err() && !dangling {
                return Err(format!(
                    "list of the missing {rel:?} succeeded with {} entries",
                    entries.len()
                ));
            }
            let via_link = std::fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink())
                && matches!(&meta, Ok(m) if m.is_dir());
            if via_link && entries.is_empty() && !want.is_empty() && cap > 0 {
                return Ok(Some("N1"));
            }
            if matches!(&meta, Ok(m) if !m.is_dir()) && !entries.is_empty() {
                return Err(format!("list of the file {rel:?} returned entries"));
            }
            if !readable {
                return Ok(None);
            }
            let mut got: Vec<(String, u8, u64)> = entries
                .iter()
                .map(|e| {
                    (
                        e["name"].as_str().unwrap_or("").to_string(),
                        match e["kind"].as_str() {
                            Some("file") => 0,
                            Some("dir") => 1,
                            _ => 2,
                        },
                        e["size"].as_u64().unwrap_or(0),
                    )
                })
                .collect();
            got.sort();
            want.sort();
            if !ctx.find_printf {
                // The busybox rung: sizes are 0 and a link to a directory is a file.
                for g in &mut got {
                    g.1 = 0;
                    g.2 = 0;
                }
                for w in &mut want {
                    w.1 = 0;
                    w.2 = 0;
                }
            }
            if want.len() <= cap {
                if got != want {
                    return Err(format!("list {rel:?}:\n got  {got:?}\n want {want:?}"));
                }
            } else {
                if got.len() != cap {
                    return Err(format!(
                        "list {rel:?} limit={cap} returned {} of {} kept entries",
                        got.len(),
                        want.len()
                    ));
                }
                if let Some(g) = got.iter().find(|g| !want.contains(g)) {
                    return Err(format!("list {rel:?} returned {g:?}, not in {want:?}"));
                }
            }
            Ok(None)
        }
        _ => Err(format!("list answered {resp}")),
    }
}
