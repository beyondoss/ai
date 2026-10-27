//! The agent's coding tools — pi's tool set, ported.
//!
//! Each tool implements [`agent_core::Tool`]; [`default_registry`] assembles the set the agent
//! advertises to the model.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_core::{Tool, ToolError, ToolRegistry};

pub mod bash;
pub mod bash_jobs;
pub mod bash_proc;
pub mod bash_sandbox;
pub mod code_mode;
pub mod edit;
pub mod encoding;
pub mod exec;
pub(crate) mod fence_tables;
pub mod find;
pub mod fs;
pub mod grep;
pub mod job_notify;
pub mod ls;
pub mod mcp;
pub mod mcp_apps;
pub mod mcp_events;
pub(crate) mod mcp_handshake;
pub mod mcp_host;
pub mod mcp_http_exit;
pub mod mcp_manifest;
pub(crate) mod mcp_oauth;
pub mod mcp_skills;
pub mod mcp_stdio;
pub(crate) mod mcp_view_http;
pub(crate) mod mcp_wire;
pub mod memory;
pub mod output;
pub mod read;
pub mod selection;
pub mod structured_output;
pub mod subagent;
pub mod todo;
pub mod tool_search;
pub mod web;
pub mod write;

/// A duration for a message a person or the model reads: whole seconds when it is (to the tenth)
/// whole ("1 second", "2 seconds", "90 seconds"), one decimal otherwise ("1.5 seconds"), and
/// "N ms" under a second. Truncating to whole seconds showed 1500 ms as "1 seconds" and 500 ms as
/// "0 seconds".
pub(crate) fn format_duration(d: std::time::Duration) -> String {
    let ms = d.as_millis();
    if ms < 1000 && ms != 0 {
        return format!("{ms} ms");
    }
    let tenths = (ms + 50) / 100;
    if tenths.is_multiple_of(10) {
        let s = tenths / 10;
        format!("{s} second{}", if s == 1 { "" } else { "s" })
    } else {
        format!("{}.{} seconds", tenths / 10, tenths % 10)
    }
}

/// Normalize a filesystem tool's `path` argument the way `bash` gets for free from the real shell it
/// spawns: expand a leading `~` or `~/` to the user's home directory, fold non-ASCII Unicode space
/// characters (a pasted non-breaking space, common when a path is copied from a terminal or rich-text
/// source) to a plain ASCII one, and strip a leading `@` (pi's own `@file` CLI convention, occasionally
/// forwarded verbatim by a caller that composed the argument from a `run --json`-style transcript).
///
/// Every filesystem tool (`read`/`write`/`edit`/`grep`/`find`/`ls`) calls this on its path/root
/// argument before any `std::fs`/walk call. Without it, `~/notes.md` behaved inconsistently within the
/// very same turn: `bash: cat ~/notes.md` succeeds (the spawned shell expands `~` itself), while
/// `read({"path": "~/notes.md"})` failed with a confusing `ENOENT` — `std::fs` has no shell and treats
/// `~` as a literal directory name.
///
/// Falls back to returning `path` unchanged if `$HOME` isn't set (or is empty) rather than erroring —
/// consistent with every other tool here, which surface a filesystem error at the point of actual use
/// rather than pre-validating input that might still resolve to something real.
pub(crate) fn normalize_path(path: &str) -> String {
    normalize_path_with_home(path, home_dir().as_deref())
}

/// [`normalize_path`], but against a caller-supplied home rather than this process's `$HOME`.
///
/// Split out because `~` does not mean the same thing on every filesystem a tool might act on. When a
/// backend's files live somewhere other than this host (see [`fs::PathWorld`]), expanding `~/notes.md`
/// against the *host's* `$HOME` produces a path for the wrong machine's user — silently, and in a form
/// that looks entirely plausible. [`expand_tilde`] already took `home: Option<&str>` for exactly this
/// reason; this is the rest of the normalization chain finally doing the same.
pub(crate) fn normalize_path_with_home(path: &str, home: Option<&str>) -> String {
    if let Some(decoded) = decode_file_url(path) {
        return decoded;
    }
    let path = path.strip_prefix('@').unwrap_or(path);
    let folded: String = path
        .chars()
        .map(|c| {
            if c != ' ' && c != '\t' && c != '\n' && c != '\r' && c.is_whitespace() {
                ' '
            } else {
                c
            }
        })
        .collect();
    expand_tilde(&folded, home)
}

/// [`normalize_path_with_home`] without the Unicode-space fold: the path as the model spelled it, so a
/// file whose real name holds U+00A0 or U+202F can be addressed exactly.
fn normalize_exact(path: &str, home: Option<&str>) -> String {
    decode_file_url(path)
        .unwrap_or_else(|| expand_tilde(path.strip_prefix('@').unwrap_or(path), home))
}

/// Unwrap a `file://` URL argument into a plain filesystem path — pi's `normalizePath` does the same
/// (`fileURLToPath`) before any other normalization, so a path a model composed from something it saw
/// as a URL (a log line, an editor deep-link) still resolves instead of every fs tool reporting a
/// confusing "not found". POSIX-only (an empty or `localhost` host, matching Node's `fileURLToPath` on
/// non-Windows) — fine for this Linux-only target. Returns `None` for anything not `file://`-prefixed,
/// so the caller falls through to the normal tilde/space-folding path unchanged.
fn decode_file_url(path: &str) -> Option<String> {
    let rest = path.strip_prefix("file://")?;
    let rest = rest.strip_prefix("localhost").unwrap_or(rest);
    Some(
        percent_encoding::percent_decode_str(rest)
            .decode_utf8_lossy()
            .into_owned(),
    )
}

fn home_dir() -> Option<String> {
    std::env::var("HOME").ok().filter(|h| !h.is_empty())
}

/// Whether `root` (or any of its ancestors) contains a `.git` entry — a lightweight, dependency-free
/// approximation of git's own repository discovery. `find`/`grep` use this to decide whether their
/// `ignore::WalkBuilder` should git-boundary-respect (inside a repo — a nested repo's own `.gitignore`
/// then can't leak parent rules across the boundary, matching pi's own `fd`/`rg` invocation, which is
/// only forced `--no-require-git` *outside* a git repo) or unconditionally honor `.gitignore` files as
/// plain ignore files (outside a repo, where there's no `.git` to anchor git-aware behavior at all).
pub(crate) fn root_is_inside_git_repo(root: &std::path::Path) -> bool {
    let mut dir = if root.is_dir() {
        Some(root)
    } else {
        root.parent()
    };
    while let Some(d) = dir {
        if d.join(".git").exists() {
            return true;
        }
        dir = d.parent();
    }
    false
}

/// Split from [`normalize_path`] so tilde-expansion logic is unit-testable without mutating the
/// process's `$HOME` — unsafe to do from a test that may run in parallel with others reading it (same
/// concern as `resources::tz_string_offset`). `home: None` (unset or empty `$HOME`) leaves `path`
/// unchanged rather than erroring, consistent with every tool here surfacing a filesystem error at the
/// point of actual use rather than pre-validating input that might still resolve to something real.
///
/// `pub` (not just used by `normalize_path` here): `skills`/`prompts` reuse it for
/// `--skill`/`--prompt-template`'s extra discovery-root paths, which previously weren't tilde-expanded
/// at all — usually masked by the shell doing it first, but not for a quoted argument or one built
/// programmatically. Also reached from the separate `agent` *binary* crate (`main.rs`'s
/// `read_file_refs`, Task #20 pi-parity fix) for an `@~/notes.md` CLI file reference — `pub(crate)`
/// would only reach modules inside this library crate itself, not the bin target that depends on it.
pub fn expand_tilde(path: &str, home: Option<&str>) -> String {
    let Some(home) = home else {
        return path.to_string();
    };
    if path == "~" {
        return home.to_string();
    }
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", home.trim_end_matches('/')),
        None => path.to_string(),
    }
}

/// Overwrite `path` atomically: write a sibling temp file, then `rename` it over the target.
/// `rename(2)` is atomic within one filesystem, so a concurrent reader — or a crash mid-write — sees
/// either the original file or the fully-written one, never a half-written file. The temp file is a
/// sibling (same directory) so the rename stays on one filesystem. A bare `std::fs::write` truncates
/// in place and would leave a partial file if the process died between truncation and the last byte.
///
/// If `path` is itself a symlink, the write goes through it to the real target instead — see
/// [`resolve_symlink_target`]'s doc comment for why. Everything below (the temp file, its permission
/// bits, the final `rename`) operates on the *resolved* path in that case, so the outer symlink itself
/// is left completely alone.
///
/// The temp file's name carries an unpredictable suffix ([`temp_suffix`]) and is opened with
/// `create_new` rather than plain `create`: a deterministic `.foo.tmp` name would let anything able to
/// plant a symlink in the same directory (another tool call, a prior turn) redirect this write to an
/// arbitrary path — plant `.foo.tmp -> ~/.ssh/authorized_keys`, then wait for a `write`/`edit` on
/// `foo`. `create_new` refuses to open through an existing path (including a symlink) instead of
/// following it, closing that TOCTOU window.
///
/// If `path` already exists, the temp file is created with its permission bits rather than the process
/// umask default — `rename(2)` swaps the whole directory entry, mode included, so a freshly-created
/// temp file would otherwise silently downgrade e.g. a `chmod 600` file to world-readable on every
/// edit. The bits are then set again on the open descriptor, because `open`'s mode passes through the
/// umask. This is [`FsBackend::write_bytes`](crate::tools::fs::FsBackend::write_bytes)'s contract,
/// which `ShellFs` keeps the same way.
///
/// Shared by `write` and `edit`: both replace whole files and must not leave a corrupt intermediate
/// state that a later read (or `serve` reattach) would observe.
///
/// **Not synced to disk**: a crash leaves the old file or the new one, but a power loss can still
/// lose the write. That is the right trade for the model's file edits, which an `fsync` would make
/// cost milliseconds each (about 9 ms on macOS) for no benefit. State that must survive a power loss
/// uses [`write_atomic_durable`].
pub(crate) fn write_atomic(path: &str, content: &[u8]) -> std::io::Result<()> {
    write_atomic_with(path, content, false)
}

/// [`write_atomic`], and durable: the temp file is `fsync`ed before the rename and the directory
/// after it, so a power loss leaves the old file or the new one, never a path naming an empty or
/// partial file. A filesystem may commit the rename (metadata) ahead of the data, which without the
/// first sync is exactly how `auth.json` or `settings.json` comes back empty. For the settings, trust,
/// auth and MCP-auth stores — small, rarely written, and costly to lose.
pub(crate) fn write_atomic_durable(path: &str, content: &[u8]) -> std::io::Result<()> {
    write_atomic_with(path, content, true)
}

fn write_atomic_with(path: &str, content: &[u8], durable: bool) -> std::io::Result<()> {
    use std::io::Write as _;

    let p = std::path::Path::new(path);
    let resolved = resolve_symlink_target(p)?;
    let p = resolved.as_path();
    // Never a fresh file renamed over a lock file — by its name in any case, by where it sits, or by
    // being one this process holds (see `file_lock::is_lock_file`, which only `stat`s: a FIFO or a
    // device is never opened, nobody's lock is ever tested): the lock would stay on the old inode,
    // which the path no longer names.
    if crate::file_lock::is_lock_file(p) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("{} is a lock file; it is not replaced", p.display()),
        ));
    }
    let name = match p.file_name() {
        Some(name) => name,
        None => return Err(std::io::Error::other(format!("invalid path: {path}"))),
    };

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    let existing = std::fs::metadata(p).ok().map(|m| m.permissions());
    #[cfg(unix)]
    if let Some(perms) = &existing {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        opts.mode(perms.mode());
    }

    let tmp = p.with_file_name(format!(".{}.tmp.{}", name.to_string_lossy(), temp_suffix()));
    let mut f = opts.open(&tmp)?;
    // `open`'s mode is filtered through the umask, so a 0664 file would come back 0644 under the usual
    // 022. Setting it on the open descriptor makes it exact, before any content is written.
    if let Some(perms) = existing
        && let Err(e) = f.set_permissions(perms)
    {
        drop(f);
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // Durable: the bytes are on disk before the rename can be (see `write_atomic_durable`).
    let written = f.write_all(content).and_then(|()| {
        if !durable {
            return Ok(());
        }
        durability_step(p, DurabilityStep::SyncTemp);
        f.sync_all()
    });
    if let Err(e) = written {
        drop(f);
        let _ = std::fs::remove_file(&tmp); // don't leave the temp behind on a mid-write failure
        return Err(e);
    }
    drop(f);

    durability_step(p, DurabilityStep::Rename);
    match std::fs::rename(&tmp, p) {
        Ok(()) => {
            if durable {
                sync_parent_dir(p);
            }
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp); // don't leave the temp behind on failure
            Err(e)
        }
    }
}

/// Make a rename in `path`'s directory durable: until the directory itself is synced, a power loss
/// can bring back the entry as it was before. Best effort, because the rename has already happened —
/// the new content is in place and readable, and reporting the write failed would be wrong. Some
/// filesystems refuse `fsync` on a directory, and off unix a directory cannot be opened to sync.
fn sync_parent_dir(path: &std::path::Path) {
    #[cfg(unix)]
    {
        let dir = match path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d,
            _ => std::path::Path::new("."),
        };
        durability_step(path, DurabilityStep::SyncDir);
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// The durability-relevant steps of [`write_atomic_durable`] (and of [`write_atomic`]), recorded in
/// order under test, with the path written, so their order can be asserted without tracing syscalls.
/// Process-wide rather than per-thread: the file tools write on the blocking pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DurabilityStep {
    SyncTemp,
    Rename,
    SyncDir,
}

#[cfg(test)]
static DURABILITY_STEPS: std::sync::Mutex<Vec<(std::path::PathBuf, DurabilityStep)>> =
    std::sync::Mutex::new(Vec::new());

#[inline]
fn durability_step(path: &std::path::Path, step: DurabilityStep) {
    #[cfg(test)]
    DURABILITY_STEPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((path.to_path_buf(), step));
    #[cfg(not(test))]
    let _ = (path, step);
}

/// Take the steps recorded for writes to `path` (other tests' writes are left alone).
#[cfg(test)]
fn take_durability_steps(path: &std::path::Path) -> Vec<DurabilityStep> {
    let mut all = DURABILITY_STEPS.lock().unwrap_or_else(|e| e.into_inner());
    let mut mine = Vec::new();
    all.retain(|(p, step)| {
        if p == path {
            mine.push(*step);
            false
        } else {
            true
        }
    });
    mine
}

/// Resolve `path` to its real target when it's a symlink, so [`write_atomic`] writes through it
/// instead of replacing the link — pi's `write`/`read` tools are built on Node's `fs.writeFile`/
/// `fs.readFile`, which open the given path directly and so *follow* a trailing symlink to its real
/// target. `write_atomic`'s own `rename(2)`-based swap does not: `rename(2)` never follows a trailing
/// symlink at its destination, it replaces the directory entry itself — so writing straight to `path`
/// when it's a symlink silently severs the link (planting a disconnected regular file at the link's
/// own path) while leaving the real target completely untouched. A symlink managed by a dotfile
/// manager (stow, chezmoi) or shared across a monorepo is exactly the case this breaks.
///
/// Follows every link, live or dangling, one `read_link` hop at a time (each relative target joined to
/// the directory of the link it came from), for at most [`MAX_SYMLINK_HOPS`] hops — the kernel's own
/// `MAXSYMLINKS`. This is `FILES-OVER-EXEC.md` section 5.3 step 2, which `ShellFs` follows too, so both
/// backends give the same answer for the same tree. A bare `canonicalize` is not enough: it refuses a
/// dangling link (the write's own target-to-create) and a loop alike, and a single-hop fallback would
/// resolve a chain of dangling links one hop short — or, on a loop, rename the temp file over the other
/// link, replacing a symlink with a regular file.
///
/// A path still a link after the last hop is a loop: `ELOOP` ("Too many levels of symbolic links (os
/// error 40)"), before anything is created. A `symlink_metadata` that fails with `ELOOP` (a path
/// through a link fan-out the kernel itself refuses) is the same error; any other failure — a dangling
/// target in a missing directory — ends the walk at that path, so the write fails as any write into a
/// missing directory does.
///
/// Returns `path` itself when it isn't a symlink at all, including "doesn't exist yet".
fn resolve_symlink_target(path: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    let mut cur = path.to_path_buf();
    for _ in 0..MAX_SYMLINK_HOPS {
        if !is_link(&cur)? {
            return Ok(cur);
        }
        let target = std::fs::read_link(&cur)?;
        cur = if target.is_absolute() {
            target
        } else {
            match cur.parent() {
                Some(dir) => dir.join(target),
                None => target,
            }
        };
    }
    if is_link(&cur)? {
        return Err(std::io::Error::from_raw_os_error(ELOOP));
    }
    Ok(cur)
}

/// Whether `path` is a symlink. Missing or unreadable is "no" (the write then fails as any write
/// there does); only the kernel's own `ELOOP` is an error.
fn is_link(path: &std::path::Path) -> std::io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(m) => Ok(m.file_type().is_symlink()),
        Err(e) if e.raw_os_error() == Some(ELOOP) => Err(e),
        Err(_) => Ok(false),
    }
}

/// Linux's `MAXSYMLINKS`, the bound on [`resolve_symlink_target`]'s walk.
const MAX_SYMLINK_HOPS: usize = 40;
/// `ELOOP`: 40 on Linux, 62 on macOS.
#[cfg(target_os = "linux")]
const ELOOP: i32 = 40;
#[cfg(not(target_os = "linux"))]
const ELOOP: i32 = 62;

/// An unpredictable per-call suffix for [`write_atomic`]'s temp file name — see
/// [`beyond_sandbox::fs::temp_suffix`].
pub(crate) use beyond_sandbox::fs::temp_suffix;

/// Canonicalize a path for use as a same-file grouping key (`Tool::write_target`): two same-turn calls
/// naming the same underlying file — `./foo.rs`, `foo.rs`, or a symlink alias — must land on one key so
/// the loop serializes them instead of racing two concurrent read-modify-writes on one file.
///
/// Real canonicalization (`std::fs::canonicalize`) resolves symlinks and `.`/`..` against the actual
/// filesystem, but fails with `NotFound` for a path that doesn't exist yet — exactly `write`'s common
/// case (creating a new file). For that case it does **not** fall back to the typed spelling: the
/// nearest existing ancestor is canonicalized and the missing tail appended
/// ([`resolve_nearest`](crate::path_utils::resolve_nearest)), so `sl/new.env` with `sl -> secrets`
/// keys as `secrets/new.env` — the key the file will have once it exists. On a case-insensitive
/// volume (probed, not assumed — [`is_case_insensitive_at`](crate::path_utils::is_case_insensitive_at))
/// the key is lowercased, so `SECRETS/new.env` and `secrets/new.env` are one file to the deny-list,
/// the approval memory and the write locks, as they are to the filesystem. The returned `bool` says
/// whether that folding happened, which a deny glob needs to know to match case-insensitively too.
///
/// Never fails outright.
///
/// `root` is the tool's configured root (see [`resolve_against`]); a relative `path` is keyed against it
/// rather than against the *process* cwd. This is what makes two subagents running in separate git
/// worktrees, each editing `src/lib.rs`, resolve to two different keys — as they must, since those are
/// two different files. Keying both against one process-wide cwd would serialize them against each other
/// for no reason, and (worse) a shared `WriteLockRegistry` would report them as the same file. An empty
/// `root` means "the process cwd", preserving the pre-root behavior exactly.
pub(crate) fn canonical_write_target_ci(root: &Path, path: &str) -> (String, bool) {
    let p = Path::new(path);
    let joined = if p.is_relative() && !root.as_os_str().is_empty() {
        root.join(p)
    } else {
        p.to_path_buf()
    };
    // `resolve_nearest` is host syscalls, so reaching here at all means the path is being keyed
    // against this host — the only world in which joining the process cwd is a true answer.
    let real = crate::path_utils::resolve_nearest(&joined);
    let ci = crate::path_utils::is_case_insensitive_at(&real);
    (crate::path_utils::comparison_key(&real, ci), ci)
}

/// [`canonical_write_target_ci`]'s key alone.
#[cfg(test)]
pub(crate) fn canonical_write_target(root: &Path, path: &str) -> String {
    canonical_write_target_ci(root, path).0
}

/// The purely lexical counterpart of [`canonical_write_target_ci`]: fold away `.` and `..` and join a
/// relative path onto `root`, touching no filesystem at all.
///
/// It exists as its own function because `canonicalize` is a **host** syscall, and calling it for a
/// path that isn't on this host is worse than useless. With a sandbox mounted at the same absolute
/// path as the working tree — the configuration we actually want, so skills and the prompt's `cwd`
/// stay true — `canonicalize` *succeeds*, resolving the **host's** symlinks, and returns a key for a
/// different file than the one about to be written. Getting the right answer by accident when paths
/// differ and the wrong one in the recommended configuration is the worst possible failure mode, so
/// the choice is made explicitly by [`write_key`] rather than left to whether a syscall happened to
/// fail.
///
/// The `world` is consulted for one thing only: an **empty root**, which means "the process cwd".
/// That is a true answer on this host and a fabricated one anywhere else — the replica's cwd is not
/// the sandbox's — so a remote path with no root stays relative rather than being rooted at a
/// directory the target has never heard of. Every other branch is pure string work, identical in
/// both worlds.
pub(crate) fn lexical_write_target(root: &Path, path: &str, world: &fs::PathWorld) -> String {
    let p = std::path::Path::new(path);
    let mut normalized = if p.is_absolute() || (root.as_os_str().is_empty() && world.is_remote()) {
        PathBuf::new()
    } else if root.as_os_str().is_empty() {
        std::env::current_dir().unwrap_or_default()
    } else {
        root.to_path_buf()
    };
    for component in p.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized.display().to_string()
}

/// The one definition of "which file is this call about".
///
/// Three separate subsystems need to answer that question and **must** answer it identically:
///
/// - `write`/`edit`'s [`Tool::write_target`](agent_core::Tool::write_target), which
///   `WriteLockRegistry` uses to serialize concurrent writes to one file;
/// - [`ToolPolicy::before_tool_call`](crate::policy::ToolPolicy), where `--deny-path` decides whether
///   a write is allowed at all;
/// - [`approval::scope_key`](crate::approval::scope_key), where an "always allow" decision is
///   remembered.
///
/// The latter two are security controls. Before this function each site open-coded the same
/// `canonical_write_target(root, &resolve_against(root, path))` composition, which worked only for as
/// long as all three copies stayed in step — and a deny-list that resolves a path differently from the
/// tool that writes it is a bypass, not a bug. One definition, three callers, no drift.
pub(crate) fn write_key(root: &Path, path: &str, world: &fs::PathWorld) -> String {
    write_key_ci(root, path, world).0
}

/// [`write_key`], plus whether the key was case-folded (see [`canonical_write_target_ci`]). Only
/// ever `true` locally: a remote volume's case rules can't be probed from here.
pub(crate) fn write_key_ci(root: &Path, path: &str, world: &fs::PathWorld) -> (String, bool) {
    let resolved = resolve_against_in(root, path, world);
    match world {
        // On this host, `canonicalize` is meaningful and valuable: it unifies `./x`, `../dir/x` and a
        // symlink to `x` (or through a symlinked parent of a not-yet-created `x`) into one key, which
        // is what stops a same-turn `edit(real)` + `write(alias)` from racing on one file.
        fs::PathWorld::Local => canonical_write_target_ci(root, &resolved),
        // Elsewhere, it can only lie. See [`lexical_write_target`].
        fs::PathWorld::Remote { .. } => (lexical_write_target(root, &resolved, world), false),
    }
}

/// Resolve a model-supplied `path` argument against a filesystem tool's configured `root`: normalize it
/// ([`normalize_path`] — `~`, `file://`, a leading `@`, folded Unicode spaces, the fold skipped when the path exists exactly as given), then, if it is still
/// *relative*, join it onto `root`.
///
/// This is the seam that makes a per-child working directory possible at all. `read`/`write`/`edit` take
/// whatever path the model names and hand it straight to `std::fs`, and `ls`/`grep`/`find` default their
/// `path` to `"."` — so before this, every relative path resolved against the one *process-wide* cwd, and
/// `std::env::set_current_dir` is not an option (it's process-global, and this crate deliberately avoids
/// it; see `bash`'s own per-call `cwd` instead).
///
/// **An empty `root` means "the process cwd"**: the path is returned still-relative and the OS resolves
/// it, exactly as before this function existed. That is `Default`'s value for every tool, so a registry
/// built without an explicit root behaves identically to the pre-root code.
///
/// This is *not* containment. An absolute path is returned untouched, so a child can still name a file
/// outside its root — the same contract Claude Code's `isolation: "worktree"` and pi's subagent `cwd`
/// offer. The guarantee is that two cooperating children fanned out in parallel don't silently stomp each
/// other, not that a determined one is jailed.
pub(crate) fn resolve_against(root: &Path, path: &str) -> String {
    let exact = join_against(root, normalize_exact(path, home_dir().as_deref()));
    let folded = join_against(root, normalize_path(path));
    // The space fold exists for pasted/typo'd paths. A file whose real name contains U+00A0 and
    // friends must still be reachable by that exact name, so the fold only applies when it isn't there.
    if exact != folded && std::fs::symlink_metadata(&exact).is_ok() {
        exact
    } else {
        folded
    }
}

/// [`resolve_against`], but expanding `~` against the home of whichever filesystem `world` names —
/// this host's `$HOME` for [`fs::PathWorld::Local`], the target's for a remote one. The join itself is
/// pure string work and identical either way; only tilde expansion is world-sensitive.
///
/// This is what every filesystem tool calls, `bash` included. [`resolve_against`] is the local world
/// spelled out, and the local arm below *is* that call, so the two can never drift.
pub(crate) fn resolve_against_in(root: &Path, path: &str, world: &fs::PathWorld) -> String {
    match world {
        fs::PathWorld::Local => resolve_against(root, path),
        fs::PathWorld::Remote { home } => {
            join_against(root, normalize_path_with_home(path, home.as_deref()))
        }
    }
}

/// Join an already-normalized path onto `root` when it is still relative. The one definition of the
/// join, shared by both worlds — it touches no filesystem and so cannot differ between them.
fn join_against(root: &Path, normalized: String) -> String {
    if root.as_os_str().is_empty() || Path::new(&normalized).is_absolute() {
        return normalized;
    }
    root.join(&normalized).display().to_string()
}

/// The refusal `read`, `write` and `edit` return for a path that is neither a regular file nor a
/// directory (a FIFO, socket or device), decided from the [`FileKind`](fs::FileKind) the backend's
/// `stat` already reported, before anything opens the path. Asking the host's `std::fs::metadata`
/// instead would answer about the wrong machine for a remote path.
///
/// The gate exists because opening one of these is a real hang: a FIFO opened for reading blocks
/// until a writer shows up (and for writing, until a reader does), and a character device like
/// `/dev/zero` opens and reads immediately but never signals EOF or contains a newline byte, so a
/// line-scanning read loop spins forever. None of this has a timeout or kill-on-drop the way
/// `bash`'s child processes do, and because it is a blocking syscall inside the future's own
/// `poll()`, cancelling the enclosing future cannot recover it — in `serve` it pins a tokio worker
/// thread shared by other sessions. There is no legitimate use for reading or writing one through
/// these tools, so all of them are refused with one clear error. A directory needs no gate: opening
/// one already fails cleanly a moment later.
///
/// `tool` names the caller in the message (`"read"`/`"write"`/`"edit"`) so the model sees which
/// tool refused the path.
pub(crate) fn non_regular_file_error(path: &str, tool: &str) -> ToolError {
    ToolError::Execution(format!(
        "{path} is not a regular file (it's a FIFO, socket, or device); `{tool}` can only operate on \
         regular files"
    ))
}

/// Whether `path` is actually writable by *this process* — a real access check, not an inspection of
/// the file's mode bits. On Unix, `std::fs::Permissions::readonly()` means "no write bit set for
/// *anyone*" (`mode & 0o222 == 0`): a file owned by a different uid/gid than this process (a build
/// artifact left by another user/container) can report `readonly() == false` — "writable" — even
/// though this process's actual credentials can't write it, so a pre-check built on it can pass right
/// before the real write fails with a generic OS error. pi's own `edit.ts` pre-check calls the real
/// `access(2)` syscall (`fsAccess(path, R_OK|W_OK)`), which the OS evaluates against the *calling
/// process's* credentials, not just the file's mode bits.
///
/// Neither `rustix` nor `nix` (the two crates that would wrap `access(2)` directly) is a dependency
/// anywhere in this workspace — checked every `Cargo.toml`, not just this crate's; `rustix` appears in
/// `Cargo.lock` only transitively, several layers deep, via other crates' own dependencies — so this
/// uses the simplest portable equivalent instead: actually open the file for write access, then
/// immediately drop the handle without writing or truncating a single byte. `OpenOptions::write(true)`
/// with `truncate`/`create` left at their defaults (`false`) makes the OS itself evaluate the *real*
/// permission check — the same uid/gid-aware answer `access(2)` would give — and closing without ever
/// calling `write` means this can't corrupt, truncate, or even touch the file's mtime.
///
/// A missing file (`NotFound`) is treated as "writable": callers that need the file to already exist
/// (`edit`) fail on that separately, with a clearer message, before ever reaching this check; a caller
/// that creates new files (`write`) must not be blocked by this pre-check just because the file isn't
/// there yet.
pub(crate) fn is_writable(path: &str) -> bool {
    match std::fs::OpenOptions::new().write(true).open(path) {
        Ok(_) => true,
        Err(e) => e.kind() == std::io::ErrorKind::NotFound,
    }
}

/// The files this registry's `read`/`write`/`edit` have touched, most recent last, bounded. `edit`
/// searches them when an `old_string` is missing from its own file, so the error can name the file the
/// text is actually in — the model's most common edit error is sending one file's text to another.
///
/// Registry-scoped: a `set_model` rebuild starts it empty, which only costs that hint until the model
/// reads or edits again. Paths are stored resolved, the form the tools do I/O with.
#[derive(Default)]
pub struct TouchedFiles(std::sync::Mutex<std::collections::VecDeque<String>>);

impl TouchedFiles {
    const CAP: usize = 32;

    pub fn record(&self, path: &str) {
        let mut q = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = q.iter().position(|p| p == path) {
            q.remove(i);
        } else if q.len() == Self::CAP {
            q.pop_front();
        }
        q.push_back(path.to_string());
    }

    /// Newest first.
    pub fn recent(&self) -> Vec<String> {
        let q = self.0.lock().unwrap_or_else(|e| e.into_inner());
        q.iter().rev().cloned().collect()
    }
}

/// Everything [`default_registry_with_config`] needs to build the tool set.
///
/// This replaces what used to be a ladder of five `default_registry_with_prefix_and_…` functions, each
/// adding one positional knob and delegating down. That convention stopped paying for itself at
/// `default_registry_with_prefix_image_auto_resize_and_mcp_tools`: a sixth knob (`root`, which subagents
/// need) would have meant a sixth name and a sixth wrapper, for two real production call sites
/// (`commands/run.rs::run_task` and `serve/build.rs::build_tools`). `Default` + struct-update syntax gives every
/// caller the same "only name what you override" ergonomics the ladder existed to provide.
#[derive(Default)]
pub struct ToolConfig<'a> {
    /// `bash` waits for every command to exit instead of leaving one quiet for
    /// [`bash::IDLE_YIELD`] running as a `bash_job` job. Only for a caller with no `bash_job` to come
    /// back with: `serve`'s host `bash` command.
    pub bash_waits_for_exit: bool,
    /// Which shell `bash` runs commands through (`--bash-shell-path`). Trusted as already validated at
    /// CLI-argument time rather than re-checked on every registry rebuild (`set_model` rebuilds).
    pub bash_shell_path: Option<&'a str>,
    /// A line prepended to every `bash` command in the same shell invocation
    /// (`--bash-command-prefix`, matching pi's `shellCommandPrefix`).
    pub bash_command_prefix: Option<&'a str>,
    /// Which shell `bash` runs commands through when [`ToolConfig::fs_backend`] is **remote**.
    ///
    /// The local resolution order (`/bin/bash`, else `bash` on `$PATH`, else `sh`) is a set of host
    /// filesystem probes, and their answer is about the wrong machine: a `/bin/bash` here says nothing
    /// about a sandbox that may well be Alpine, where there is none. So in the remote world the host is
    /// never probed and this value is used instead — whoever stood the sandbox up is the one who can
    /// ask it (`command -v bash`). Unset falls back to `sh`, which every POSIX target has, rather than
    /// to a `bash` a busybox image would fail every single command on.
    ///
    /// Ignored in the local world, and ignored anywhere when `bash_shell_path` is set: an explicit
    /// operator override outranks both.
    pub remote_shell: Option<&'a str>,
    /// Whether `read` downscales an oversized image to fit the inline budget (pi's
    /// `ImageSettings.autoResize`). `Default` is `false` here but every constructor below sets it
    /// `true` — see [`ToolConfig::new`], which is the only way this struct should be built.
    pub image_auto_resize: bool,
    /// The directory every filesystem tool resolves a *relative* path against, and `bash`'s default
    /// `cwd`. **Empty means "the process cwd"** — the pre-root behavior — which is what `default_registry`
    /// and every test that doesn't care about roots gets. A subagent running under `isolation: worktree`
    /// passes its worktree here. See [`resolve_against`].
    pub root: PathBuf,
    /// When true, MCP tools are not advertised as `mcp__…` entries. They become the deferred catalog
    /// inside the Code Mode `execute` tool (`tools.<server>.<method>`). Off by default. A no-op
    /// unless this binary was built with `--features code-mode` — otherwise MCP stays advertised
    /// (hiding the catalog behind an `execute` that does not exist would strand it). The CLI rejects
    /// `--code-mode` on a density-default binary before this constructor runs.
    pub code_mode: bool,
    /// How MCP tools reach the model when Code Mode is off: deferred behind `tool_search`, eager, or
    /// (the default) whichever the catalog's size makes cheaper. See [`tool_search`].
    pub mcp_loading: tool_search::McpLoading,
    /// `--exclude-tools` names applied to the nested Code Mode catalog (so an excluded MCP tool is
    /// unreachable through `execute` as well as missing from the advertised set).
    pub nested_exclude: &'a [String],
    /// `--deny-tool` names applied to the nested catalog — same names the policy hook blocks.
    pub nested_deny: &'a [String],
    /// Already-connected tools from configured MCP servers (see [`mcp::connect_all`]), registered after
    /// every built-in so an allow/deny filter scopes them too.
    ///
    /// Already-resolved, not a list of servers to connect: connecting is async (a process spawn / HTTP
    /// connection plus a `tools/list` round trip) while this whole construction path is deliberately
    /// synchronous (see `tools::mcp`'s module doc). Each call site connects once at startup and passes
    /// the resulting `Arc`s in on every rebuild, so a `serve` rebuild reuses live connections rather
    /// than reconnecting to every server.
    pub mcp_tools: &'a [Arc<dyn Tool>],
    /// `--web-allow-private`: let the `web` tool reach loopback/private/link-local addresses. Off by
    /// default — the tool refuses them to prevent SSRF, since it fetches URLs the model chose. See
    /// [`web`].
    pub web_allow_private: bool,
    /// `--web-allow-host`: specific hostnames the `web` tool may reach even when private egress is off
    /// (an internal service, or `127.0.0.1` for a test fixture).
    pub web_allow_hosts: &'a [String],
    /// `--web-timeout-ms`: the `web` tool's per-request timeout (default 30 s).
    pub web_timeout_ms: Option<u64>,
    /// Where `bash` runs.
    ///
    /// `None` means this host. A `Some` runner **must** be the same target `fs_backend` points at:
    /// a `bash` on one machine and an `edit` on another is not a configuration, it is a bug — the
    /// model would write a file and then run a command that cannot see it. More importantly, `bash`
    /// is the tool that most needs containment, so isolating the filesystem while leaving arbitrary
    /// command execution on the host would provide close to no isolation at all.
    /// [`default_registry_with_config`] is the only place that pairs them, precisely so they cannot
    /// drift apart.
    pub command_runner: Option<Arc<dyn exec::CommandRunner>>,
    /// Where the six filesystem tools' I/O lands.
    ///
    /// `None` means the host filesystem — every existing caller, and behaviorally identical to the
    /// code before this seam existed. A `Some` backend is shared by `Arc` across all six tools and
    /// across every registry rebuild, the same lifecycle [`ToolConfig::mcp_tools`] has and for the
    /// same reason: the registry is rebuilt on every `set_model`, and a backend that reconnected per
    /// rebuild would be a bug rather than a cost.
    pub fs_backend: Option<Arc<dyn fs::FsBackend>>,
    /// Discovered skills, so `read` resolves the `skill:<name>` paths the system prompt's listing
    /// names them by ([`crate::skills::SKILL_SCHEME`]). Empty: the scheme names nothing.
    pub skills: &'a [crate::skills::Skill],
    /// The job table `bash` parks commands in (see [`bash_jobs`]). `Some` for a host that owns the
    /// jobs past one registry — `serve`'s session (so `set_model` keeps them, and the idle session
    /// can wake on them), `run` (which stops them at the end and says which). `None`: a table of
    /// this registry's own, dropped (killing its jobs) with it — a subagent's.
    pub bash_jobs: Option<Arc<bash_jobs::BashJobs>>,
}

impl<'a> ToolConfig<'a> {
    /// A config with the production defaults: image auto-resize on, root = the process cwd, no bash
    /// overrides, no MCP tools. Prefer this over `ToolConfig::default()` — `image_auto_resize` derives
    /// to `false`, which is *not* the default the CLI or `serve` want.
    pub fn new() -> Self {
        Self {
            image_auto_resize: true,
            ..Self::default()
        }
    }

    /// Set the filesystem root every tool resolves relative paths against. See [`ToolConfig::root`].
    pub fn with_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.root = root.into();
        self
    }
}

/// Built-ins that [`default_registry`] constructs but a bare `run`/`serve` does **not** advertise.
///
/// `web` stays constructed so `--tools web` can opt it in, but it is not a coding-agent default: a
/// TB polyglot run burned most of its `$` fetching pages. `grep`/`find`/`ls`/`todo` stay advertised.
pub const OPTIONAL_BUILTINS: &[&str] = &["web"];

/// The constructed tool set: Pi's seven coding tools (read, write, edit, bash, ls, grep, find),
/// `bash_job` (whenever `bash` can leave a command running), the `todo` task list, and `web`. The
/// advertised default is all of them minus `web`, which stays registered so `--tools web` can opt
/// it in. No vendor-specific platform tools, no root (process cwd).
pub fn default_registry() -> ToolRegistry {
    default_registry_with_config(&ToolConfig::new())
}

/// Assemble the tool set described by `cfg`. The one real constructor; everything else here is a
/// shorthand over it.
pub fn default_registry_with_config(cfg: &ToolConfig<'_>) -> ToolRegistry {
    let root = cfg.root.as_path();
    let mut reg = ToolRegistry::new();
    // One backend, shared by all six filesystem tools — so they cannot disagree about which
    // filesystem they are acting on, and a `read` can always see what a `write` just wrote.
    let touched = Arc::new(TouchedFiles::default());
    let mut read_tool = read::Read::new(root)
        .with_image_auto_resize(cfg.image_auto_resize)
        .with_skills(cfg.skills)
        .with_touched(touched.clone());
    let mut write_tool = write::Write::new(root).with_touched(touched.clone());
    let mut edit_tool = edit::Edit::new(root).with_touched(touched);
    let mut ls_tool = ls::Ls::new(root);
    let mut grep_tool = grep::Grep::new(root);
    let mut find_tool = find::Find::new(root);
    if let Some(b) = &cfg.fs_backend {
        read_tool = read_tool.with_backend(b.clone());
        write_tool = write_tool.with_backend(b.clone());
        edit_tool = edit_tool.with_backend(b.clone());
        ls_tool = ls_tool.with_backend(b.clone());
        grep_tool = grep_tool.with_backend(b.clone());
        find_tool = find_tool.with_backend(b.clone());
    }
    reg.register(Arc::new(read_tool));
    reg.register(Arc::new(write_tool));
    reg.register(Arc::new(edit_tool));
    reg.register(Arc::new(ls_tool));
    reg.register(Arc::new(grep_tool));
    reg.register(Arc::new(find_tool));
    // `bash` runs wherever the filesystem tools do — see `ToolConfig::command_runner`.
    let mut bash = match &cfg.command_runner {
        Some(runner) => bash::Bash::with_runner(runner.clone()),
        None => bash::Bash::real(),
    };
    bash = bash.with_root(root);
    // `bash` asks the filesystem backend which world it is in, rather than being told once, because
    // `ExecCell` can be re-pointed mid-session (`set_exec_endpoint`) with no registry rebuild — a
    // captured world would keep resolving `~` and validating `cwd` against the machine the session
    // just left. Only when a `command_runner` is present too: that pairing is what makes the backend's
    // world `bash`'s world (see [`ToolConfig::command_runner`]), and without it `bash` really is on
    // this host however remote the files are.
    if cfg.command_runner.is_some()
        && let Some(b) = &cfg.fs_backend
    {
        bash = bash.with_backend(b.clone());
    }
    if let Some(path) = cfg.bash_shell_path {
        bash = bash.with_shell_path(path);
    }
    if let Some(shell) = cfg.remote_shell {
        bash = bash.with_remote_shell(shell);
    }
    if let Some(prefix) = cfg.bash_command_prefix {
        bash = bash.with_command_prefix(prefix);
    }
    if cfg.bash_waits_for_exit {
        bash = bash.with_idle_yield(None);
    }
    if let Some(jobs) = &cfg.bash_jobs {
        bash = bash.with_jobs(jobs.clone());
    }
    let bash = Arc::new(bash);
    reg.register(bash.clone());
    // Over the same instance, so `bash_job` sees the jobs this `bash` parks.
    if bash.detaches() {
        reg.register(Arc::new(bash::BashJob::new(bash)));
    }
    // Stateless: it validates and echoes the model's own full-replace list. See `todo`'s module doc for
    // why it holds nothing across calls (this registry is rebuilt on every `set_model`).
    reg.register(Arc::new(todo::Todo::new()));
    // Owns its own reqwest client (SSRF resolver, no default redirects) — see `web`'s module doc. The
    // egress policy is fixed at build time from `cfg`; a `set_model` rebuild reconstructs it, which is
    // fine (the policy is cheap and immutable for the process).
    reg.register(Arc::new(web::Web::new(
        cfg.web_allow_private,
        cfg.web_allow_hosts,
        cfg.web_timeout_ms,
    )));
    // After every built-in, so an allow/deny filter (`apply_filter`) scopes MCP tools too, and so a
    // server can't shadow a built-in by name (the `mcp__<server>__<tool>` prefix already prevents that).
    // Code Mode defers that catalog: MCP tools stay callable from `execute` but are not advertised.
    // Without `--features code-mode` there is no interpreter to hide them behind, so they stay direct.
    // Otherwise `cfg.mcp_loading` picks: deferred behind `tool_search` (no interpreter) or eager.
    #[cfg(feature = "code-mode")]
    if cfg.code_mode {
        let nested =
            code_mode::select_deferred_tools(cfg.mcp_tools, cfg.nested_exclude, cfg.nested_deny);
        if let Some(execute) = code_mode::execute_tool(nested) {
            reg.register(execute);
        }
    } else {
        tool_search::register_mcp(&mut reg, cfg.mcp_tools, cfg.mcp_loading);
    }
    #[cfg(not(feature = "code-mode"))]
    {
        let _ = cfg.code_mode;
        tool_search::register_mcp(&mut reg, cfg.mcp_tools, cfg.mcp_loading);
    }
    reg
}

/// Restrict a registry to an allow-list, a deny-list, or nothing at all — the CLI/RPC surface for
/// scoping an agent's capabilities (e.g. a read-only reviewer with no `bash`/`edit`/`write`), which
/// otherwise has no way to run with less than the full default tool set. `no_tools` wins outright
/// (checked first); otherwise `tools` (if given) narrows to exactly those names, and `exclude` (if
/// given) drops names from whatever remains — so `tools` + `exclude` can combine to carve one tool out
/// of an allow-list, though picking just one of the three is the common case. Names match exactly;
/// `run`/`serve` refuse an unknown one at startup ([`selection::check_tool_flags`]), so by the time a
/// list reaches here every name is a built-in or an `mcp__` name.
///
/// With no `--tools` allow-list, [`OPTIONAL_BUILTINS`] (`web`) is dropped so a bare invocation does
/// not advertise `web`. Pass `--tools web` to opt it in; it stays constructed in [`default_registry`]
/// so the allow-list can name it.
pub fn apply_filter(
    reg: &mut ToolRegistry,
    tools: Option<&[String]>,
    exclude: Option<&[String]>,
    no_tools: bool,
) {
    // A deferred MCP tool rides with `tool_search`: an allow-list that keeps `tool_search` keeps the
    // catalog behind it (minus `--exclude-tools`), as `--tools read,bash` drops both — the same
    // reach an allow-list has over eagerly loaded MCP tools. `tool_search::refresh` then rebuilds
    // its catalog over what is left.
    let keeps_search = filter_allows(tool_search::NAME, tools, exclude, no_tools)
        && tools.is_some_and(|allow| allow.iter().any(|a| a == tool_search::NAME));
    let deferred: std::collections::HashSet<String> = reg
        .deferred_tools()
        .iter()
        .map(|t| t.name().to_string())
        .collect();
    reg.retain(|name| {
        filter_allows(name, tools, exclude, no_tools)
            || (keeps_search
                && deferred.contains(name)
                && filter_allows(name, None, exclude, no_tools))
    });
    tool_search::refresh(reg);
}

/// How a registry build places MCP tools — the part of a build [`rebase_mcp`] has to repeat when a
/// server's tools change under a running loop (see [`mcp::tool_source`]).
#[derive(Clone, Debug, Default)]
pub struct McpFilter {
    pub loading: tool_search::McpLoading,
    /// `--tools`.
    pub tools: Option<Vec<String>>,
    /// `--exclude-tools`.
    pub exclude: Option<Vec<String>>,
    /// `--no-tools`.
    pub no_tools: bool,
    /// `--code-mode`: MCP tools live inside `execute`, which a rebase does not rebuild.
    pub code_mode: bool,
}

/// `current` with its MCP tools (every `mcp__` name, and `tool_search` over them) replaced by
/// `mcp_tools`, placed the way a fresh build places them — eager or deferred by `filter.loading`, then
/// filtered by `filter`'s allow/deny lists. Everything else in `current` (built-ins, `subagent`,
/// `structured_output`, `memory`) is kept as it is: those were registered past the filter on purpose.
pub fn rebase_mcp(
    current: &ToolRegistry,
    mcp_tools: &[Arc<dyn Tool>],
    filter: &McpFilter,
) -> ToolRegistry {
    let mut mcp = ToolRegistry::new();
    tool_search::register_mcp(&mut mcp, mcp_tools, filter.loading);
    apply_filter(
        &mut mcp,
        filter.tools.as_deref(),
        filter.exclude.as_deref(),
        filter.no_tools,
    );
    let mut next = current.clone();
    next.retain(|name| !name.starts_with("mcp__") && name != tool_search::NAME);
    for name in mcp.names() {
        let Some(tool) = mcp.get(&name) else { continue };
        if mcp.is_deferred(&name) {
            next.register_deferred(tool);
        } else {
            next.register(tool);
        }
    }
    next
}

/// Whether [`apply_filter`] keeps a tool named `name` — the one predicate, so a caller that runs a
/// tool outside the registry (an MCP App view's `tools/call`) applies exactly the model's filter.
pub fn filter_allows(
    name: &str,
    tools: Option<&[String]>,
    exclude: Option<&[String]>,
    no_tools: bool,
) -> bool {
    if no_tools {
        return false;
    }
    // `bash_job` waits on what `bash` started, so it is kept or dropped with `bash`.
    let name = if name == bash::JOB_TOOL { "bash" } else { name };
    if tools.is_some_and(|allow| !allow.iter().any(|a| a == name)) {
        return false;
    }
    if exclude.is_some_and(|deny| deny.iter().any(|d| d == name)) {
        return false;
    }
    // No `--tools` allow-list: drop opt-in built-ins (`web`). `--tools web` is how it comes back;
    // `--exclude-tools` alone must not re-advertise it.
    !(tools.is_none() && OPTIONAL_BUILTINS.contains(&name))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::tools::fs::FsBackend as _;

    use super::*;

    #[test]
    fn format_duration_is_exact_about_fractions() {
        use std::time::Duration as D;
        for (d, want) in [
            (D::from_millis(0), "0 seconds"),
            (D::from_millis(500), "500 ms"),
            (D::from_millis(900), "900 ms"),
            (D::from_secs(1), "1 second"),
            (D::from_millis(1500), "1.5 seconds"),
            (D::from_secs(2), "2 seconds"),
            (D::from_secs(90), "90 seconds"),
            (D::from_millis(1960), "2 seconds"),
            (D::from_millis(90_500), "90.5 seconds"),
        ] {
            assert_eq!(format_duration(d), want, "{d:?}");
        }
    }

    #[test]
    fn root_is_inside_git_repo_detects_a_git_dir_at_the_root_itself() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        assert!(root_is_inside_git_repo(dir.path()));
    }

    #[test]
    fn root_is_inside_git_repo_detects_a_git_dir_in_an_ancestor() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let nested = dir.path().join("src").join("deep");
        std::fs::create_dir_all(&nested).unwrap();
        assert!(root_is_inside_git_repo(&nested));
    }

    #[test]
    fn root_is_inside_git_repo_is_false_with_no_git_dir_anywhere() {
        let dir = crate::test_support::isolated_tempdir();
        let nested = dir.path().join("src");
        std::fs::create_dir_all(&nested).unwrap();
        assert!(!root_is_inside_git_repo(&nested));
    }

    #[test]
    fn expand_tilde_expands_bare_tilde_to_home() {
        assert_eq!(expand_tilde("~", Some("/home/jared")), "/home/jared");
    }

    #[test]
    fn expand_tilde_expands_tilde_slash_prefix() {
        assert_eq!(
            expand_tilde("~/notes.md", Some("/home/jared")),
            "/home/jared/notes.md"
        );
    }

    #[test]
    fn expand_tilde_does_not_touch_a_tilde_in_the_middle_of_a_path() {
        // Only a *leading* `~`/`~/` is a home-directory reference — `foo/~bar` is a literal filename.
        assert_eq!(expand_tilde("foo/~bar", Some("/home/jared")), "foo/~bar");
    }

    #[test]
    fn expand_tilde_leaves_path_unchanged_when_home_is_unavailable() {
        assert_eq!(expand_tilde("~/notes.md", None), "~/notes.md");
    }

    #[test]
    fn expand_tilde_trims_a_trailing_slash_on_home_before_joining() {
        assert_eq!(
            expand_tilde("~/notes.md", Some("/home/jared/")),
            "/home/jared/notes.md"
        );
    }

    #[test]
    fn normalize_path_strips_a_leading_at_prefix() {
        assert_eq!(normalize_path("@/etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn normalize_path_folds_a_non_breaking_space_to_a_plain_space() {
        // U+00A0 NBSP — a common artifact of copy-pasting a path from a terminal or rich-text source.
        assert_eq!(normalize_path("foo\u{00A0}bar.txt"), "foo bar.txt");
    }

    #[test]
    fn resolve_against_prefers_an_existing_file_with_a_unicode_space_in_its_name() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("shot\u{202F}1.txt");
        std::fs::write(&real, "x").unwrap();
        let sibling = dir.path().join("shot 1.txt");
        // Exists exactly as given: addressed exactly, not folded onto a sibling name.
        assert_eq!(
            resolve_against(dir.path(), "shot\u{202F}1.txt"),
            real.display().to_string()
        );
        // Does not exist as given: the typo fold still applies.
        assert_eq!(
            resolve_against(dir.path(), "shot\u{00A0}1.txt"),
            sibling.display().to_string()
        );
    }

    #[test]
    fn normalize_path_is_a_no_op_for_an_ordinary_absolute_path() {
        assert_eq!(normalize_path("/etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn normalize_path_unwraps_a_file_url() {
        // pi-parity gap (fixed): every fs tool rejected `file://`-prefixed paths outright, unlike
        // pi's `normalizePath` (`fileURLToPath`) — a model that composed a path from something it saw
        // as a URL (a log line, an editor deep-link) got a confusing "not found" from all six tools.
        assert_eq!(normalize_path("file:///etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn normalize_path_unwraps_a_file_url_with_an_explicit_localhost_host() {
        assert_eq!(normalize_path("file://localhost/etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn normalize_path_percent_decodes_a_file_url() {
        assert_eq!(
            normalize_path("file:///home/jared/my%20notes.md"),
            "/home/jared/my notes.md"
        );
    }

    #[test]
    fn normalize_path_file_url_bypasses_tilde_and_at_handling() {
        // A `file://` URL's path is already absolute and already decoded — it isn't a `~`-prefixed or
        // `@`-prefixed argument, so those normalizations must not additionally apply to it.
        assert_eq!(
            normalize_path("file:///home/jared/~weird@name.txt"),
            "/home/jared/~weird@name.txt"
        );
    }

    /// The durable write syncs the new content before the rename publishes it, and the directory
    /// after: a power loss can leave the old file or the new one, never a path naming an empty or
    /// partial file. A failed rename syncs no directory.
    #[test]
    fn write_atomic_durable_syncs_the_temp_before_the_rename_and_the_directory_after() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("auth.json");
        let steps = || take_durability_steps(&target);
        write_atomic_durable(target.to_str().unwrap(), b"{}").unwrap();
        #[cfg(unix)]
        let expected = [
            DurabilityStep::SyncTemp,
            DurabilityStep::Rename,
            DurabilityStep::SyncDir,
        ];
        #[cfg(not(unix))]
        let expected = [DurabilityStep::SyncTemp, DurabilityStep::Rename];
        assert_eq!(steps(), expected);
        assert_eq!(std::fs::read(&target).unwrap(), b"{}");

        let blocked = dir.path().join("a-directory");
        std::fs::create_dir(&blocked).unwrap();
        assert!(write_atomic_durable(blocked.to_str().unwrap(), b"x").is_err());
        assert_eq!(
            take_durability_steps(&blocked),
            [DurabilityStep::SyncTemp, DurabilityStep::Rename],
            "a rename that failed published nothing to sync"
        );
    }

    /// The `write` and `edit` tools never pay for an `fsync`: their path (`LocalFs` → `write_atomic`)
    /// renames without syncing anything.
    #[tokio::test]
    async fn the_write_and_edit_tools_do_not_fsync() {
        use crate::tools::fs::FsBackend as _;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("main.rs");
        let fs = crate::tools::fs::local::LocalFs::new();
        fs.write_bytes(&target, b"fn main() {}".to_vec())
            .await
            .unwrap();
        let mtime = std::fs::metadata(&target).unwrap().modified().unwrap();
        assert!(
            fs.write_if_unchanged(&target, b"fn main() { }".to_vec(), Some(mtime))
                .await
                .unwrap()
        );
        assert_eq!(
            take_durability_steps(&target),
            [DurabilityStep::Rename, DurabilityStep::Rename],
            "a tool write renamed into place and synced nothing"
        );
    }

    #[test]
    fn write_atomic_cleans_up_temp_on_rename_failure() {
        let dir = tempfile::tempdir().unwrap();
        // A regular file can't be `rename`d over an existing directory (EISDIR) — forces the rename
        // failure path without relying on permissions or a second filesystem.
        let target = dir.path().join("target");
        std::fs::create_dir(&target).unwrap();

        assert!(write_atomic(target.to_str().unwrap(), b"new content").is_err());

        // The sibling temp file must not survive a failed rename — `write_atomic` removes it
        // explicitly on the failure path rather than leaving litter behind. `target` (the directory)
        // should be the only entry left.
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(
            entries.len(),
            1,
            "failed rename left extra files behind: {:?}",
            entries.iter().map(|e| e.file_name()).collect::<Vec<_>>()
        );
        // The original target is untouched by the failed write.
        assert!(target.is_dir());
    }

    /// Never a fresh file renamed over a lock file — the record one or, during mixed rollout, the
    /// legacy one an older binary locks — directly or through a symlink: the lock would be left on
    /// an inode the path no longer names. An unrelated `*.lock` (a `Cargo.lock`) is written as ever.
    #[test]
    #[cfg(unix)]
    fn write_atomic_never_replaces_a_lock_file() {
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("s1");
        std::fs::create_dir_all(&session).unwrap();
        let _held = crate::file_lock::try_lock(crate::file_lock::Target::Dir(&session))
            .unwrap()
            .unwrap();
        let legacy = session.join("lock");
        let link = dir.path().join("innocent.txt");
        std::os::unix::fs::symlink(&legacy, &link).unwrap();
        let before = std::fs::metadata(&legacy).unwrap();
        for target in [&legacy, &link] {
            assert!(
                write_atomic(target.to_str().unwrap(), b"x").is_err(),
                "{} must not be replaced",
                target.display()
            );
        }
        use std::os::unix::fs::MetadataExt as _;
        assert_eq!(std::fs::metadata(&legacy).unwrap().ino(), before.ino());
        let cargo_lock = dir.path().join("Cargo.lock");
        write_atomic(cargo_lock.to_str().unwrap(), b"[[package]]\n").unwrap();
        assert_eq!(std::fs::read(&cargo_lock).unwrap(), b"[[package]]\n");
    }

    /// The lock files a held record lock file beside them cannot show: an old binary's session
    /// directory (a segment and its legacy `lock`, no record lock file), its session file's
    /// `<f>.lock`, a journal key held through `Target::Itself`, and a record lock file spelled in
    /// another case. A `lock` in a directory that holds no session is an ordinary file.
    #[test]
    #[cfg(unix)]
    fn write_atomic_never_replaces_a_lock_file_a_record_beside_it_cannot_show() {
        let dir = tempfile::tempdir().unwrap();
        let old_session = dir.path().join("old");
        std::fs::create_dir_all(&old_session).unwrap();
        std::fs::write(old_session.join("000001.jsonl"), "").unwrap();
        let legacy = old_session.join("lock");
        std::fs::write(&legacy, "").unwrap();
        assert!(
            write_atomic(legacy.to_str().unwrap(), b"x").is_err(),
            "an old binary's session lock is not replaced"
        );
        let old_file_lock = dir.path().join("1700000000_abc.jsonl.lock");
        std::fs::write(&old_file_lock, "").unwrap();
        assert!(write_atomic(old_file_lock.to_str().unwrap(), b"x").is_err());
        let key = dir.path().join("journal.key");
        let held = crate::file_lock::try_lock(crate::file_lock::Target::Itself(&key))
            .unwrap()
            .unwrap();
        assert!(
            write_atomic(key.to_str().unwrap(), b"x").is_err(),
            "a key held through Itself is not replaced"
        );
        drop(held);
        write_atomic(key.to_str().unwrap(), b"x").unwrap();
        let shouty = dir.path().join("S2.BEYOND-LOCK");
        assert!(write_atomic(shouty.to_str().unwrap(), b"x").is_err());
        let plain = dir.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        write_atomic(plain.join("lock").to_str().unwrap(), b"x").unwrap();
    }

    /// Deciding whether a target is a lock file never opens it: a FIFO with no writer (whose
    /// read-only open blocks forever) is written at once, and a device is only `stat`ed.
    #[test]
    #[cfg(unix)]
    fn write_atomic_never_opens_its_target_to_decide() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("pipe");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .is_ok_and(|s| s.success());
        assert!(made, "mkfifo");
        let (tx, rx) = std::sync::mpsc::channel();
        let path = fifo.to_str().unwrap().to_owned();
        std::thread::spawn(move || {
            let _ = tx.send(write_atomic(&path, b"x"));
        });
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("write_atomic hung on a FIFO")
            .unwrap();
        assert_eq!(std::fs::read(&fifo).unwrap(), b"x");
        for device in ["/dev/null", "/dev/zero", "/dev/tty"] {
            assert!(!crate::file_lock::is_lock_file(std::path::Path::new(
                device
            )));
        }
    }

    /// A SQLite database in WAL mode that another program holds open mid-transaction (its POSIX
    /// locks on the file) is an ordinary file, not a lock file: an edit of it goes through.
    #[test]
    #[cfg(unix)]
    fn write_atomic_edits_a_sqlite_database_another_program_holds() {
        use std::io::{BufRead as _, BufReader};
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("app.db");
        let Ok(mut other) = std::process::Command::new("python3")
            .args([
                "-c",
                "import sqlite3,sys\nc=sqlite3.connect(sys.argv[1],isolation_level=None)\nc.execute('pragma journal_mode=wal')\nc.execute('create table t(x)')\nc.execute('begin immediate')\nc.execute('insert into t values(1)')\nprint('held',flush=True)\nsys.stdin.read()",
            ])
            .arg(&db)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
        else {
            eprintln!("no python3: skipping the SQLite check");
            return;
        };
        let mut line = String::new();
        BufReader::new(other.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        assert_eq!(line.trim(), "held");
        let written = write_atomic(db.to_str().unwrap(), b"replaced");
        drop(other.stdin.take());
        other.wait().unwrap();
        written.unwrap();
    }

    /// On a case-insensitive filesystem (macOS; a casefold ext4 directory) `LOCK` and `.BEYOND-LOCK`
    /// are the held lock files themselves, and `LOCK` in an old binary's session directory is its
    /// legacy lock. Skipped where the temp dir is case-sensitive.
    #[test]
    #[cfg(unix)]
    fn write_atomic_never_replaces_a_lock_file_through_a_folded_spelling() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Probe"), "").unwrap();
        if !dir.path().join("PROBE").exists() {
            return;
        }
        let session = dir.path().join("s1");
        std::fs::create_dir_all(&session).unwrap();
        let _held = crate::file_lock::try_lock(crate::file_lock::Target::Dir(&session))
            .unwrap()
            .unwrap();
        let old_session = dir.path().join("old");
        std::fs::create_dir_all(&old_session).unwrap();
        std::fs::write(old_session.join("000001.jsonl"), "").unwrap();
        std::fs::write(old_session.join("lock"), "").unwrap();
        for folded in [
            session.join("LOCK"),
            session.join(".BEYOND-LOCK"),
            old_session.join("LOCK"),
        ] {
            assert!(
                write_atomic(folded.to_str().unwrap(), b"x").is_err(),
                "{} reaches a lock file",
                folded.display()
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn write_atomic_does_not_follow_a_symlink_planted_at_the_old_deterministic_temp_name() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("secret.txt");
        std::fs::write(&target, b"original").unwrap();

        // Simulate a symlink preplanted at the old, guessable `.{name}.tmp` path, pointing
        // somewhere the write must never reach.
        let decoy = dir.path().join("decoy-target");
        let planted = dir.path().join(".secret.txt.tmp");
        std::os::unix::fs::symlink(&decoy, &planted).unwrap();

        write_atomic(target.to_str().unwrap(), b"new content").unwrap();

        assert!(!decoy.exists(), "write_atomic followed the planted symlink");
        assert_eq!(std::fs::read_link(&planted).unwrap(), decoy);
        assert_eq!(std::fs::read(&target).unwrap(), b"new content");
    }

    #[test]
    #[cfg(unix)]
    fn resolve_follows_a_chain_of_dangling_links_across_directories() {
        use std::os::unix::fs::symlink;
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        std::fs::create_dir_all(r.join("a")).unwrap();
        std::fs::create_dir_all(r.join("b")).unwrap();
        symlink("../b/l2", r.join("a/l1")).unwrap();
        symlink("../a/l3", r.join("b/l2")).unwrap();
        symlink("../a/final.txt", r.join("a/l3")).unwrap();
        symlink("a/l1", r.join("l0")).unwrap();
        let resolved = resolve_symlink_target(&r.join("l0")).unwrap();
        assert_eq!(resolved.file_name().unwrap(), "final.txt");
        assert!(!resolved.exists(), "the chain ends in a missing file");
        write_atomic(r.join("l0").to_str().unwrap(), b"x").unwrap();
        assert_eq!(std::fs::read(r.join("a/final.txt")).unwrap(), b"x");
        assert!(
            std::fs::symlink_metadata(r.join("l0"))
                .unwrap()
                .is_symlink()
        );
    }

    #[test]
    #[cfg(unix)]
    fn resolve_follows_an_absolute_dangling_target() {
        let d = tempfile::tempdir().unwrap();
        let abs = d.path().join("made.txt");
        std::os::unix::fs::symlink(&abs, d.path().join("l")).unwrap();
        assert_eq!(resolve_symlink_target(&d.path().join("l")).unwrap(), abs);
    }

    #[test]
    #[cfg(unix)]
    fn resolve_returns_a_non_link_path_unchanged() {
        let d = tempfile::tempdir().unwrap();
        for p in [d.path().join("missing"), d.path().to_path_buf()] {
            assert_eq!(resolve_symlink_target(&p).unwrap(), p);
        }
    }

    #[test]
    #[cfg(unix)]
    fn resolve_fails_a_loop_with_eloop_and_write_changes_nothing() {
        use std::os::unix::fs::symlink;
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        symlink("b", r.join("a")).unwrap();
        symlink("a", r.join("b")).unwrap();
        symlink("self", r.join("self")).unwrap();
        for name in ["a", "b", "self"] {
            let err = resolve_symlink_target(&r.join(name)).unwrap_err();
            assert_eq!(err.raw_os_error(), Some(ELOOP), "{name}");
            let err = write_atomic(r.join(name).to_str().unwrap(), b"x").unwrap_err();
            assert!(err.to_string().contains("os error 40"), "{err}");
        }
        assert_eq!(
            std::fs::read_link(r.join("a")).unwrap(),
            std::path::Path::new("b")
        );
        assert_eq!(
            std::fs::read_link(r.join("b")).unwrap(),
            std::path::Path::new("a")
        );
        assert_eq!(std::fs::read_dir(r).unwrap().count(), 3, "no temp left");
    }

    #[test]
    #[cfg(unix)]
    fn resolve_allows_exactly_forty_hops() {
        use std::os::unix::fs::symlink;
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        // l0 -> l1 -> ... -> l40 -> end.txt: 41 links is one too many, 40 is fine.
        for n in [40usize, 41] {
            let sub = r.join(format!("c{n}"));
            std::fs::create_dir(&sub).unwrap();
            for i in 0..n {
                symlink(format!("l{}", i + 1), sub.join(format!("l{i}"))).unwrap();
            }
            symlink("end.txt", sub.join(format!("l{n}"))).unwrap();
        }
        // From c40/l1: 40 reads reach end.txt. From c41/l0: 41 links, still a link after 40 hops.
        let ok = resolve_symlink_target(&r.join("c40/l1"));
        assert_eq!(ok.unwrap(), r.join("c40/end.txt"));
        let err = resolve_symlink_target(&r.join("c41/l0")).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(ELOOP));
    }

    #[test]
    #[cfg(unix)]
    fn resolve_fails_a_link_fan_out_fast() {
        use std::os::unix::fs::symlink;
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        symlink(".", r.join("l0")).unwrap();
        for i in 1..=12 {
            let p = format!("l{}", i - 1);
            symlink(format!("{p}/{p}/{p}"), r.join(format!("l{i}"))).unwrap();
        }
        let err = write_atomic(r.join("l12").to_str().unwrap(), b"x").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(ELOOP));
        assert!(
            std::fs::symlink_metadata(r.join("l12"))
                .unwrap()
                .is_symlink()
        );
    }

    #[test]
    #[cfg(unix)]
    fn write_through_a_dangling_link_into_a_missing_directory_is_not_found() {
        let d = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink("nodir/f.txt", d.path().join("l")).unwrap();
        let err = write_atomic(d.path().join("l").to_str().unwrap(), b"x").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert_eq!(std::fs::read_dir(d.path()).unwrap().count(), 1);
    }

    #[test]
    #[cfg(unix)]
    fn write_atomic_writes_through_a_symlink_leaving_it_intact() {
        // Empirically verified bug (pi-parity pass 15): `rename(2)` never follows a trailing symlink
        // at its destination — it replaces the directory entry itself — so writing straight to a
        // symlinked path silently severed the link (a fresh regular file landed at the link's own
        // path) while the real target went untouched. pi's `write`/`read` tools are built on Node's
        // `fs.writeFile`/`fs.readFile`, which open the path directly and so *do* follow a trailing
        // symlink — this proves `write_atomic` now matches that: the real target's content changes,
        // and the outer path is still a genuine symlink afterward.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.txt");
        std::fs::write(&real, b"original").unwrap();
        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        write_atomic(link.to_str().unwrap(), b"updated via the symlink").unwrap();

        assert_eq!(
            std::fs::read(&real).unwrap(),
            b"updated via the symlink",
            "the real target's content must change"
        );
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the outer path must still be a symlink afterward, not replaced by a regular file"
        );
        assert_eq!(
            std::fs::read_link(&link).unwrap(),
            real,
            "the symlink must still point at the same real target"
        );
    }

    #[test]
    #[cfg(unix)]
    fn write_atomic_preserves_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("secret.env");
        std::fs::write(&target, b"SECRET=1").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();

        write_atomic(target.to_str().unwrap(), b"SECRET=2").unwrap();

        let mode = std::fs::metadata(&target).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "write_atomic silently changed the file's permissions"
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"SECRET=2");
    }

    #[test]
    fn canonical_write_target_unifies_spellings_of_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap().join("foo.rs");
        std::fs::write(&real, b"fn main() {}").unwrap();

        let dotted = dir.path().join("./foo.rs");
        let via_parent = dir.path().join("sub/../foo.rs");
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();

        let a = canonical_write_target(Path::new(""), real.to_str().unwrap());
        let b = canonical_write_target(Path::new(""), dotted.to_str().unwrap());
        let c = canonical_write_target(Path::new(""), via_parent.to_str().unwrap());
        assert_eq!(a, b, "absolute vs `./`-prefixed spelling must match");
        assert_eq!(a, c, "a path through `..` must resolve to the same key");
    }

    #[test]
    #[cfg(unix)]
    fn canonical_write_target_unifies_a_symlink_with_its_real_target() {
        // A same-turn `edit(real_path)` and `write(symlink_to_real_path)` must serialize under one
        // write-lock key — otherwise they'd run fully concurrently against the same underlying file
        // (pi: file-mutation-queue.test.ts, "uses the same queue for symlink aliases").
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap().join("target.txt");
        std::fs::write(&real, b"hello").unwrap();
        let link = dir.path().join("alias.txt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let via_real = canonical_write_target(Path::new(""), real.to_str().unwrap());
        let via_symlink = canonical_write_target(Path::new(""), link.to_str().unwrap());
        assert_eq!(
            via_real, via_symlink,
            "a symlink and its real target must canonicalize to the same write-lock key"
        );
    }

    #[test]
    fn canonical_write_target_falls_back_to_lexical_normalization_when_the_file_is_new() {
        // `write` routinely targets a path that doesn't exist yet — `canonicalize` fails (NotFound),
        // so the grouping key must still unify spellings without touching the filesystem. Uses
        // absolute paths (rather than `std::env::set_current_dir`, which is process-global and unsafe
        // to mutate from a test that may run in parallel with others) so the relative-path branch of
        // `canonical_write_target` isn't exercised here, only the `.`/`..` lexical resolution.
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();

        let a = canonical_write_target(Path::new(""), base.join("brand-new.rs").to_str().unwrap());
        let b =
            canonical_write_target(Path::new(""), base.join("./brand-new.rs").to_str().unwrap());
        let c = canonical_write_target(
            Path::new(""),
            base.join("sub/../brand-new.rs").to_str().unwrap(),
        );

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert!(
            std::path::Path::new(&a).is_absolute(),
            "fallback must still produce an absolute key: {a}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_new_file_through_a_symlinked_parent_shares_one_write_lock_key() {
        // two same-turn writes creating one file — one via `sl/`, one via the real directory —
        // must serialize. Falling back to the typed spelling gave them two keys.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("real")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("real"), dir.path().join("sl")).unwrap();
        let world = fs::PathWorld::Local;
        assert_eq!(
            write_key(dir.path(), "sl/new.rs", &world),
            write_key(dir.path(), "real/new.rs", &world)
        );
    }

    #[test]
    fn default_registry_has_the_coding_tools_and_nothing_vendor_specific() {
        let reg = default_registry();
        // pi's coding tools …
        for name in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert!(reg.get(name).is_some(), "missing coding tool: {name}");
        }
        // … `bash`'s companion for a command still running after its call returned …
        assert!(reg.get(bash::JOB_TOOL).is_some(), "missing bash_job");
        // … the model's own task list, and its window to the web (constructed, not advertised).
        assert!(reg.get("todo").is_some(), "missing todo tool");
        assert!(reg.get("web").is_some(), "missing web tool");
        // Nothing vendor-specific: the default set is the general coding toolset and nothing else.
        // Platform tools belong in whatever ships them, not in the agent everyone runs.
        for name in ["fork", "sync", "logs"] {
            assert!(
                reg.get(name).is_none(),
                "vendor-specific tool leaked in: {name}"
            );
        }
        assert_eq!(reg.len(), 10);
    }

    #[test]
    fn advertised_default_omits_web() {
        let mut reg = default_registry();
        apply_filter(&mut reg, None, None, false);
        for name in [
            "read", "write", "edit", "bash", "bash_job", "ls", "grep", "find", "todo",
        ] {
            assert!(reg.get(name).is_some(), "missing advertised tool: {name}");
        }
        assert!(
            reg.get("web").is_none(),
            "web must be opt-in via --tools, not advertised by default"
        );
        assert_eq!(reg.len(), 9);
    }

    #[test]
    fn tools_allow_list_opts_in_web() {
        let mut reg = default_registry();
        apply_filter(&mut reg, Some(&["read".into(), "web".into()]), None, false);
        assert!(reg.get("web").is_some());
        assert!(reg.get("read").is_some());
        assert!(reg.get("grep").is_none());
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn apply_filter_no_tools_wins_outright() {
        let mut reg = default_registry();
        apply_filter(
            &mut reg,
            Some(&["read".to_string()]),
            Some(&["write".to_string()]),
            true,
        );
        assert!(reg.is_empty());
    }

    #[test]
    fn apply_filter_allow_list_keeps_only_named_tools() {
        let mut reg = default_registry();
        apply_filter(
            &mut reg,
            Some(&["read".to_string(), "ls".to_string()]),
            None,
            false,
        );
        assert_eq!(reg.len(), 2);
        assert!(reg.get("read").is_some());
        assert!(reg.get("ls").is_some());
        assert!(reg.get("bash").is_none());
    }

    #[test]
    fn apply_filter_empty_allow_list_disables_every_tool() {
        // pi: regression #2835, "disables all tools when the allowlist is empty" — an empty `tools:
        // []` array (distinct from `no_tools`/omitting the field entirely) must still empty the
        // registry via the ordinary allow-list path: `retain(|name| [].iter().any(|a| a == name))` is
        // `false` for every name, same end state as `no_tools`, but reached through a different branch
        // than `apply_filter_no_tools_wins_outright` exercises.
        let mut reg = default_registry();
        apply_filter(&mut reg, Some(&[]), None, false);
        assert!(reg.is_empty(), "expected zero tools, got {}", reg.len());
    }

    #[test]
    fn apply_filter_deny_list_drops_named_tools() {
        let mut reg = default_registry();
        apply_filter(
            &mut reg,
            None,
            Some(&["bash".to_string(), "edit".to_string(), "write".to_string()]),
            false,
        );
        assert!(reg.get("bash").is_none());
        assert!(reg.get("edit").is_none());
        assert!(reg.get("write").is_none());
        assert!(reg.get("read").is_some());
        assert!(reg.get("grep").is_some());
        assert!(reg.get("web").is_none());
        // Advertised default (8) minus bash/edit/write.
        assert_eq!(reg.len(), 5);
    }

    #[test]
    fn apply_filter_allow_and_deny_combine() {
        let mut reg = default_registry();
        // Allow read/write/edit, then carve `write` back out — the intersection.
        apply_filter(
            &mut reg,
            Some(&["read".to_string(), "write".to_string(), "edit".to_string()]),
            Some(&["write".to_string()]),
            false,
        );
        assert_eq!(reg.len(), 2);
        assert!(reg.get("read").is_some());
        assert!(reg.get("edit").is_some());
        assert!(reg.get("write").is_none());
    }

    #[test]
    fn apply_filter_unknown_names_are_silently_ignored() {
        let mut reg = default_registry();
        apply_filter(&mut reg, None, Some(&["does-not-exist".to_string()]), false);
        // Unknown deny names are ignored; the advertised default (nine tools, no web) still applies.
        assert_eq!(reg.len(), 9);
        assert!(reg.get("read").is_some());
        assert!(reg.get("grep").is_some());
        assert!(reg.get("web").is_none());
    }

    /// A trivial stand-in for an [`mcp::McpTool`] — that type is crate-private and always backed by a
    /// live connection, so tests exercising `ToolConfig::mcp_tools` (which only care that *some*
    /// `Arc<dyn Tool>` merges in under its own name) use this instead of standing up a real MCP
    /// server.
    struct FakeMcpTool(&'static str);
    #[async_trait::async_trait]
    impl Tool for FakeMcpTool {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "stand-in for an MCP-discovered tool"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object" })
        }
        async fn run(&self, _: serde_json::Value) -> Result<agent_core::ToolOutput, ToolError> {
            Ok(self.0.into())
        }
    }

    #[test]
    fn default_registry_with_config_merges_the_mcp_tools_in() {
        let mcp_tools: Vec<Arc<dyn Tool>> =
            vec![Arc::new(FakeMcpTool("mcp__filesystem__read_file"))];
        let reg = default_registry_with_config(&ToolConfig {
            mcp_tools: &mcp_tools,
            ..ToolConfig::new()
        });
        // Every built-in is still there …
        for name in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert!(reg.get(name).is_some(), "missing built-in tool: {name}");
        }
        // … plus the MCP-discovered one, under its already-namespaced name.
        assert!(reg.get("mcp__filesystem__read_file").is_some());
        assert_eq!(reg.len(), default_registry().len() + 1);
    }

    /// A rebase is the build it replaces: the same registry a fresh build over the new MCP tools
    /// makes — a deferred catalog updated, an eager tool registered, the filter applied — with what
    /// was registered past the filter kept.
    #[test]
    fn rebase_mcp_matches_a_fresh_build_over_the_new_tools() {
        let before: Vec<Arc<dyn Tool>> = vec![Arc::new(FakeMcpTool("mcp__s__old"))];
        let after: Vec<Arc<dyn Tool>> = vec![
            Arc::new(FakeMcpTool("mcp__s__new")),
            Arc::new(FakeMcpTool("mcp__s__hidden")),
        ];
        let exclude = vec!["mcp__s__hidden".to_string()];
        for loading in [
            tool_search::McpLoading::Search,
            tool_search::McpLoading::Eager,
        ] {
            let build = |mcp_tools: &[Arc<dyn Tool>]| {
                let mut reg = default_registry_with_config(&ToolConfig {
                    mcp_tools,
                    mcp_loading: loading,
                    ..ToolConfig::new()
                });
                apply_filter(&mut reg, None, Some(&exclude), false);
                reg
            };
            let mut current = build(&before);
            current.register(Arc::new(FakeMcpTool("structured_output")));
            let filter = McpFilter {
                loading,
                exclude: Some(exclude.clone()),
                ..McpFilter::default()
            };
            let rebased = rebase_mcp(&current, &after, &filter);
            let mut fresh = build(&after);
            fresh.register(Arc::new(FakeMcpTool("structured_output")));
            assert_eq!(rebased.names(), fresh.names(), "{loading:?}");
            assert_eq!(rebased.definitions(), fresh.definitions(), "{loading:?}");
            assert_eq!(
                rebased.deferred_definitions(),
                fresh.deferred_definitions(),
                "{loading:?}"
            );
            assert!(rebased.get("mcp__s__old").is_none());
            assert!(rebased.get("mcp__s__hidden").is_none());
        }
    }

    #[cfg(not(feature = "code-mode"))]
    #[test]
    fn code_mode_without_the_runtime_does_not_hide_mcp() {
        let mcp_tools: Vec<Arc<dyn Tool>> =
            vec![Arc::new(FakeMcpTool("mcp__filesystem__read_file"))];
        let deny: Vec<String> = Vec::new();
        let reg = default_registry_with_config(&ToolConfig {
            mcp_tools: &mcp_tools,
            code_mode: true,
            nested_deny: &deny,
            ..ToolConfig::new()
        });
        assert!(
            reg.get("execute").is_none(),
            "default binaries must not register execute (QuickJS is not linked)"
        );
        assert!(
            reg.get("mcp__filesystem__read_file").is_some(),
            "without an interpreter, Code Mode must not strand the MCP catalog"
        );
    }

    #[cfg(feature = "code-mode")]
    #[test]
    fn code_mode_defers_mcp_tools_behind_execute() {
        let mcp_tools: Vec<Arc<dyn Tool>> =
            vec![Arc::new(FakeMcpTool("mcp__filesystem__read_file"))];
        let exclude = vec!["mcp__filesystem__read_file".to_string()];
        let deny: Vec<String> = Vec::new();
        let reg = default_registry_with_config(&ToolConfig {
            mcp_tools: &mcp_tools,
            code_mode: true,
            nested_exclude: &[],
            nested_deny: &deny,
            ..ToolConfig::new()
        });
        assert!(reg.get("execute").is_some());
        assert!(
            reg.get("mcp__filesystem__read_file").is_none(),
            "MCP tools must not be advertised when Code Mode is on"
        );
        // Nested exclude drops the tool from execute's catalog too — here the only one, so there is
        // nothing left for execute to reach and it is not advertised.
        let reg = default_registry_with_config(&ToolConfig {
            mcp_tools: &mcp_tools,
            code_mode: true,
            nested_exclude: &exclude,
            nested_deny: &deny,
            ..ToolConfig::new()
        });
        assert!(reg.get("execute").is_none());
        assert!(reg.get("mcp__filesystem__read_file").is_none());
    }

    #[cfg(feature = "code-mode")]
    #[test]
    fn code_mode_with_an_empty_deferred_catalog_registers_no_execute() {
        // An `execute` whose JS context reaches no tools costs ~270 schema tokens per request for
        // nothing: no MCP servers means the registry is exactly the plain default.
        let deny: Vec<String> = Vec::new();
        let mut reg = default_registry_with_config(&ToolConfig {
            code_mode: true,
            nested_deny: &deny,
            ..ToolConfig::new()
        });
        assert!(reg.get("execute").is_none());
        assert_eq!(reg.len(), default_registry().len());
        // Nor does the post-`--tools` restore put one back.
        apply_filter(&mut reg, Some(&["read".to_string()]), None, false);
        code_mode::restore_execute(&mut reg, Vec::new(), None, false);
        assert!(reg.get("execute").is_none());
    }

    #[cfg(feature = "code-mode")]
    #[test]
    fn code_mode_restore_keeps_execute_on_a_builtins_allow_list() {
        let mcp_tools: Vec<Arc<dyn Tool>> =
            vec![Arc::new(FakeMcpTool("mcp__filesystem__read_file"))];
        let deny: Vec<String> = Vec::new();
        let mut reg = default_registry_with_config(&ToolConfig {
            mcp_tools: &mcp_tools,
            code_mode: true,
            nested_deny: &deny,
            ..ToolConfig::new()
        });
        apply_filter(&mut reg, Some(&["read".to_string()]), None, false);
        assert!(reg.get("execute").is_none());
        let nested = code_mode::select_deferred_tools(&mcp_tools, &[], &deny);
        code_mode::restore_execute(&mut reg, nested, None, false);
        assert!(reg.get("read").is_some());
        assert!(reg.get("execute").is_some());
        assert!(reg.get("mcp__filesystem__read_file").is_none());
    }

    #[test]
    fn default_registry_with_config_with_no_servers_matches_plain_default() {
        let reg = default_registry_with_config(&ToolConfig::new());
        assert_eq!(reg.len(), default_registry().len());
    }

    #[test]
    fn tool_config_new_enables_image_auto_resize_but_derived_default_does_not() {
        // `ToolConfig::default()` exists only so `..ToolConfig::new()` struct-update syntax works; it
        // derives `image_auto_resize: false`, which is *not* what any production caller wants. Anyone
        // reaching for `default()` directly would silently disable image downscaling.
        assert!(ToolConfig::new().image_auto_resize);
        assert!(!ToolConfig::default().image_auto_resize);
    }

    #[test]
    fn tool_config_defaults_to_an_empty_root_meaning_the_process_cwd() {
        assert!(ToolConfig::new().root.as_os_str().is_empty());
        assert_eq!(
            ToolConfig::new().with_root("/tmp/wt").root,
            PathBuf::from("/tmp/wt")
        );
    }

    #[test]
    fn resolve_against_joins_a_relative_path_onto_the_root() {
        assert_eq!(
            resolve_against(Path::new("/wt"), "src/lib.rs"),
            "/wt/src/lib.rs"
        );
        // `ls`/`grep`/`find` all default their `path` argument to this.
        assert_eq!(resolve_against(Path::new("/wt"), "."), "/wt/.");
    }

    #[test]
    fn resolve_against_leaves_an_absolute_path_alone() {
        // Deliberate: a worktree is conflict avoidance, not containment. A child naming an absolute
        // path outside its root reaches it, exactly as under Claude Code's `isolation: "worktree"`.
        assert_eq!(
            resolve_against(Path::new("/wt"), "/etc/hosts"),
            "/etc/hosts"
        );
    }

    #[test]
    fn resolve_against_with_an_empty_root_is_exactly_normalize_path() {
        // The pre-root behavior, which every `default_registry()` caller and every test relies on: the
        // path is left relative and the OS resolves it against the process cwd.
        for input in [
            "src/lib.rs",
            ".",
            "/etc/hosts",
            "@/etc/hosts",
            "file:///etc/hosts",
        ] {
            assert_eq!(
                resolve_against(Path::new(""), input),
                normalize_path(input),
                "empty root must not change {input:?}"
            );
        }
    }

    #[test]
    fn resolve_against_still_applies_tilde_and_file_url_normalization_before_joining() {
        // A `~`-prefixed path expands to an absolute home path, so it must NOT then be joined onto the
        // root — otherwise `~/notes.md` under a worktree would become `<wt>/home/me/notes.md`.
        let home = std::env::var("HOME").unwrap();
        assert_eq!(
            resolve_against(Path::new("/wt"), "~/notes.md"),
            format!("{home}/notes.md")
        );
        assert_eq!(
            resolve_against(Path::new("/wt"), "file:///etc/hosts"),
            "/etc/hosts"
        );
        // A leading `@` is stripped, and what remains is still relative — so it joins onto the root.
        assert_eq!(
            resolve_against(Path::new("/wt"), "@notes.md"),
            "/wt/notes.md"
        );
    }

    #[test]
    fn canonical_write_target_keys_the_same_relative_path_under_two_roots_differently() {
        // The property the whole worktree design rests on: two subagents in separate worktrees editing
        // `src/lib.rs` are editing two *different files*, so they must get two different write-lock
        // keys. Keying both against one process-wide cwd would collapse them into one.
        let tmp = tempfile::tempdir().unwrap();
        let wt_a = tmp.path().join("a");
        let wt_b = tmp.path().join("b");
        std::fs::create_dir_all(&wt_a).unwrap();
        std::fs::create_dir_all(&wt_b).unwrap();

        let key_a = canonical_write_target(&wt_a, &resolve_against(&wt_a, "src/lib.rs"));
        let key_b = canonical_write_target(&wt_b, &resolve_against(&wt_b, "src/lib.rs"));
        assert_ne!(key_a, key_b);
        assert!(key_a.ends_with("/a/src/lib.rs"), "{key_a}");
        assert!(key_b.ends_with("/b/src/lib.rs"), "{key_b}");
    }

    #[test]
    fn canonical_write_target_still_unifies_two_spellings_within_one_root() {
        // …while, within a single root, differently-spelled references to the same not-yet-created file
        // must still collapse to one key, or two same-turn `edit`s would race.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let plain = canonical_write_target(&root, &resolve_against(&root, "new.rs"));
        let dotted = canonical_write_target(&root, &resolve_against(&root, "./new.rs"));
        let via_parent = canonical_write_target(&root, &resolve_against(&root, "sub/../new.rs"));
        assert_eq!(plain, dotted);
        assert_eq!(plain, via_parent);
    }

    #[tokio::test]
    async fn a_rooted_write_lands_inside_the_root_not_the_process_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write::Write::new(root)
            .run(json!({ "path": "notes.md", "content": "hello" }))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("notes.md")).unwrap(),
            "hello"
        );
        // And nothing was created relative to the process cwd.
        assert!(!Path::new("notes.md").exists());
    }

    #[tokio::test]
    async fn a_rooted_read_resolves_relative_to_the_root() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "content-in-root").unwrap();
        let out = read::Read::new(tmp.path())
            .run(json!({ "path": "a.txt" }))
            .await
            .unwrap();
        assert!(out.text.contains("content-in-root"), "{}", out.text);
    }

    #[tokio::test]
    async fn a_rooted_ls_lists_the_root_when_path_is_omitted() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("marker.txt"), "").unwrap();
        // `path` omitted → defaults to "." → resolves to the root, not the process cwd.
        let out = ls::Ls::new(tmp.path()).run(json!({})).await.unwrap();
        assert!(out.text.contains("marker.txt"), "{}", out.text);
    }

    #[test]
    fn is_writable_is_true_for_an_ordinary_writable_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "x").unwrap();
        assert!(is_writable(path.to_str().unwrap()));
    }

    #[test]
    fn is_writable_is_true_for_a_nonexistent_path() {
        // `write` creates new files; this pre-check must not block that just because nothing is
        // there yet.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.txt");
        assert!(is_writable(path.to_str().unwrap()));
    }

    #[test]
    fn is_writable_is_false_for_a_file_with_no_write_bit_for_anyone() {
        // The ordinary case `Permissions::readonly()` also catches — kept as a baseline so a future
        // change to `is_writable` can't silently regress the common case while only fixing the uid
        // edge case below.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("locked.txt");
        std::fs::write(&path, "x").unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&path, perms).unwrap();

        assert!(!is_writable(path.to_str().unwrap()));

        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn is_writable_is_false_when_the_owner_triad_denies_write_even_though_another_triad_allows_it()
    {
        // Pi-parity fix (task 50): `Permissions::readonly()` on Unix checks whether *any* of the three
        // permission triads (owner/group/other) has a write bit set (`mode & 0o222 == 0`) — but the
        // kernel's real access decision for a given caller only ever consults *one* triad: the owner
        // triad, in full, whenever the caller owns the file (this test process does, having just
        // created it), never falling through to group/other bits just because the owner triad denies.
        // Mode `026` (owner: none; group: write; other: read+write) is exactly that mismatch: `026 &
        // 0o222 == 0o022 != 0`, so the old bits-only check reported "writable", while the real access
        // decision (owner triad is all zero) denies it — the same class of gap as pi's own scenario (a
        // file whose *effective* permission for this process doesn't match what its raw mode bits
        // suggest), reproduced here via the owner/other split instead of a cross-process uid mismatch
        // (which a test can't set up without already running as root).
        //
        // Root (and anything else with `CAP_DAC_OVERRIDE`, common in a container-based CI runner)
        // bypasses this decision entirely — detected via a `mode 0` probe file (unreadable/unwritable
        // by absolutely anyone if DAC is actually enforced) rather than guessed from `$USER`/a uid
        // syscall, and skipped in that case since the scenario genuinely can't be demonstrated there.
        let dir = tempfile::tempdir().unwrap();
        use std::os::unix::fs::PermissionsExt;

        let probe = dir.path().join("probe.txt");
        std::fs::write(&probe, "x").unwrap();
        std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o000)).unwrap();
        let dac_bypassed = std::fs::read(&probe).is_ok();
        std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o600)).unwrap();
        if dac_bypassed {
            return;
        }

        let path = dir.path().join("f.txt");
        std::fs::write(&path, "x").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o026)).unwrap();
        let old_readonly = std::fs::metadata(&path).unwrap().permissions().readonly();
        assert!(
            !old_readonly,
            "sanity: the old mode-bits-only check would have reported this file as writable"
        );

        let result = is_writable(path.to_str().unwrap());
        // Restore so the tempdir can clean itself up on drop regardless of the assertion below.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(
            !result,
            "the owner triad denies write; the real access check must reflect that even though \
             another triad's bits happen to allow it"
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_fifo_classifies_as_other_and_the_shared_error_names_the_calling_tool() {
        // Direct unit test for the gate `write.rs`/`edit.rs` now call before their own blocking opens
        // (pi-parity pass 20) — the async, `mkfifo`-driven regression tests in each tool's own test
        // module prove the *hang* is fixed; this proves the shared helper's error text actually names
        // whichever tool called it, not a generic one-size-fits-all wording. A bare `metadata()` call
        // never blocks on a FIFO (only *opening* one does), so this needs no timeout wrapper.
        let dir = tempfile::tempdir().unwrap();
        let fifo_path = dir.path().join("a.fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo_path)
            .status()
            .expect("mkfifo must be available to run this test");
        assert!(status.success(), "mkfifo failed to create the test fixture");

        // The classification itself now lives in the backend's `stat`, which is what the tools branch
        // on — so the test asserts both halves: that a FIFO is classified `Other`, and that the shared
        // error text names the calling tool. A bare `stat` never blocks on a FIFO (only *opening* one
        // does), so this needs no timeout wrapper.
        let kind = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                fs::local::LocalFs::new()
                    .stat(&fifo_path)
                    .await
                    .unwrap()
                    .unwrap()
                    .kind
            });
        assert_eq!(kind, fs::FileKind::Other, "a FIFO must classify as Other");

        let err = non_regular_file_error(fifo_path.to_str().unwrap(), "edit");
        match err {
            ToolError::Execution(msg) => {
                assert!(msg.contains("`edit`"), "got: {msg}");
                assert!(msg.contains("FIFO"), "got: {msg}");
            }
            other => panic!("expected Execution, got {other:?}"),
        }
    }

    /// `LocalFs::stat` is what every tool's "is this safe to open" branch reads, so its classification
    /// of the three ordinary cases is pinned here — the FIFO gate decides on that
    /// classification.
    fn stat_kind(path: &std::path::Path) -> Option<fs::FileKind> {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async { fs::local::LocalFs::new().stat(path).await.unwrap() })
            .map(|m| m.kind)
    }

    #[test]
    fn stat_classifies_a_regular_file_as_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "x").unwrap();
        assert_eq!(stat_kind(&path), Some(fs::FileKind::File));
    }

    #[test]
    fn stat_reports_nothing_for_a_missing_path() {
        // A missing path is left to the caller's own real open call to produce a clearer, tool-specific
        // "not found"/"not writable" error rather than duplicating that here under different wording.
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(stat_kind(&dir.path().join("does-not-exist.txt")), None);
    }

    #[test]
    fn stat_classifies_a_directory_as_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(stat_kind(dir.path()), Some(fs::FileKind::Dir));
    }

    // ---- write_key ---------------------------------------------------------------------------
    //
    // The table below is the whole point of extracting `lexical_write_target`. Every row is a place
    // where `canonicalize` and lexical resolution can disagree, which is exactly where a deny-list, a
    // remembered approval, and a write lock would stop agreeing with each other if the three call
    // sites ever drifted apart again.

    use crate::tools::fs::PathWorld;

    #[test]
    fn write_key_local_still_matches_the_old_composition_exactly() {
        // The refactor's safety net: `write_key(.., Local)` must equal the exact expression all three
        // call sites open-coded before it existed. If this ever fails, the extraction changed behavior.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.md"), "x").unwrap();
        let root = dir.path();
        for spelling in ["notes.md", "./notes.md", "sub/../notes.md", "notes.md"] {
            let before = canonical_write_target(root, &resolve_against(root, spelling));
            let after = write_key(root, spelling, &PathWorld::Local);
            assert_eq!(before, after, "spelling {spelling:?}");
        }
    }

    #[test]
    fn write_key_with_no_root_does_not_root_a_remote_path_at_this_hosts_cwd() {
        // An empty root means "the process cwd", which is a true answer here and a fabricated one
        // anywhere else. Rooting a sandbox path at the replica's cwd would hand `--deny-path` and the
        // approval scope a key for a file on the wrong machine — and one that changes if the replica
        // is ever started from a different directory.
        let here = std::env::current_dir().unwrap().display().to_string();
        let remote = write_key(
            std::path::Path::new(""),
            "notes.md",
            &PathWorld::Remote { home: None },
        );
        assert_eq!(remote, "notes.md");
        assert!(!remote.starts_with(&here));
        // Unchanged on this host, where the cwd is exactly what an empty root means.
        assert_eq!(
            write_key(std::path::Path::new(""), "notes.md", &PathWorld::Local),
            format!("{here}/notes.md"),
        );
    }

    #[test]
    fn write_key_folds_equivalent_spellings_to_one_key_in_both_worlds() {
        // `./x` and `sub/../x` are the same file however you resolve them — this must hold remotely
        // too, or a `--deny-path` glob is sidesteppable by respelling the path.
        let root = std::path::Path::new("/workspace");
        for world in [PathWorld::Local, PathWorld::Remote { home: None }] {
            let direct = write_key(root, "/workspace/notes.md", &world);
            let dotted = write_key(root, "/workspace/./notes.md", &world);
            let dotdot = write_key(root, "/workspace/sub/../notes.md", &world);
            assert_eq!(direct, dotted, "{world:?}");
            assert_eq!(direct, dotdot, "{world:?}");
        }
    }

    #[test]
    fn write_key_resolves_a_relative_path_against_the_root_not_the_process_cwd() {
        // What keeps two subagents in separate worktrees, each editing `src/lib.rs`, on two keys.
        let a = write_key(
            std::path::Path::new("/wt/a"),
            "src/lib.rs",
            &PathWorld::Remote { home: None },
        );
        let b = write_key(
            std::path::Path::new("/wt/b"),
            "src/lib.rs",
            &PathWorld::Remote { home: None },
        );
        assert_eq!(a, "/wt/a/src/lib.rs");
        assert_ne!(a, b, "two worktrees must not share one write-lock key");
    }

    #[test]
    fn write_key_local_unifies_a_symlink_with_its_target_but_remote_does_not() {
        // The single most important row. Locally, `canonicalize` collapses an alias onto the real file
        // so a same-turn `edit(real)` + `write(alias)` serialize on one lock. Remotely the same call
        // would resolve the *host's* symlinks for a file that isn't on this host — so it must not, and
        // the two spellings stay distinct rather than silently naming some other machine's file.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.txt");
        let alias = dir.path().join("alias.txt");
        std::fs::write(&real, "x").unwrap();
        std::os::unix::fs::symlink(&real, &alias).unwrap();
        let root = dir.path();

        let local_real = write_key(root, real.to_str().unwrap(), &PathWorld::Local);
        let local_alias = write_key(root, alias.to_str().unwrap(), &PathWorld::Local);
        assert_eq!(
            local_real, local_alias,
            "locally, an alias and its target are one file and must share a key"
        );

        let remote_alias = write_key(
            root,
            alias.to_str().unwrap(),
            &PathWorld::Remote { home: None },
        );
        assert_eq!(
            remote_alias,
            alias.display().to_string(),
            "a remote key must be the path as named, never a host-canonicalized rewrite"
        );
    }

    #[test]
    fn write_key_expands_tilde_against_the_worlds_own_home() {
        // A leading `~` means a different user's directory on a different machine. Expanding it
        // against this process's `$HOME` for a remote path produces an entirely plausible path for the
        // wrong home — the kind of wrong answer that never surfaces as an error.
        let root = std::path::Path::new("");
        let remote = write_key(
            root,
            "~/notes.md",
            &PathWorld::Remote {
                home: Some("/home/guest".to_string()),
            },
        );
        assert_eq!(remote, "/home/guest/notes.md");
    }

    #[test]
    fn write_key_leaves_tilde_alone_when_the_remote_home_is_unknown() {
        // Guessing is worse than not expanding: an unexpanded `~` fails loudly at the real open call,
        // while a guessed one silently reads or writes the wrong file.
        let key = write_key(
            std::path::Path::new(""),
            "~/notes.md",
            &PathWorld::Remote { home: None },
        );
        assert!(
            key.contains('~'),
            "an unknown home must leave `~` untouched, got {key:?}"
        );
    }

    #[test]
    fn normalize_path_with_home_still_handles_file_urls_and_at_prefixes() {
        // The world-aware entry point must not have lost the rest of the normalization chain.
        assert_eq!(
            normalize_path_with_home("file:///tmp/a%20b.txt", None),
            "/tmp/a b.txt"
        );
        assert_eq!(normalize_path_with_home("@/tmp/x.txt", None), "/tmp/x.txt");
        // A pasted non-breaking space folds to a plain one.
        assert_eq!(
            normalize_path_with_home("/tmp/a\u{a0}b.txt", None),
            "/tmp/a b.txt"
        );
    }
}
