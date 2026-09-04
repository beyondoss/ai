//! The agent's coding tools — pi's tool set, ported.
//!
//! Each tool implements [`agent_core::Tool`]; [`default_registry`] assembles the set the agent
//! advertises to the model. The Beyond platform tools (fork/sync/logs) register here too once added.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_core::{Tool, ToolError, ToolRegistry};

pub mod bash;
pub mod code_mode;
pub mod edit;
pub mod exec;
pub mod find;
pub mod fs;
pub mod grep;
pub mod ls;
pub mod mcp;
pub mod mcp_host;
pub mod mcp_manifest;
pub mod memory;
pub mod output;
pub mod read;
pub mod structured_output;
pub mod subagent;
pub mod todo;
pub mod web;
pub mod write;

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
/// programmatically. Also reached from the separate `beyond-ai-agent` *binary* crate (`main.rs`'s
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
/// edit.
///
/// Shared by `write` and `edit`: both replace whole files and must not leave a corrupt intermediate
/// state that a later read (or `serve` reattach) would observe.
pub(crate) fn write_atomic(path: &str, content: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;

    let p = std::path::Path::new(path);
    let resolved = resolve_symlink_target(p);
    let p = resolved.as_deref().unwrap_or(p);
    let name = match p.file_name() {
        Some(name) => name,
        None => return Err(std::io::Error::other(format!("invalid path: {path}"))),
    };

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        if let Ok(meta) = std::fs::metadata(p) {
            opts.mode(meta.permissions().mode());
        }
    }

    let tmp = p.with_file_name(format!(".{}.tmp.{}", name.to_string_lossy(), temp_suffix()));
    let mut f = opts.open(&tmp)?;
    if let Err(e) = f.write_all(content) {
        drop(f);
        let _ = std::fs::remove_file(&tmp); // don't leave the temp behind on a mid-write failure
        return Err(e);
    }
    drop(f);

    match std::fs::rename(&tmp, p) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp); // don't leave the temp behind on failure
            Err(e)
        }
    }
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
/// Returns `None` when `path` isn't a symlink at all — including "doesn't exist yet", `write`'s
/// ordinary new-file case — so the caller's existing direct-write behavior is unchanged in every case
/// but an actual symlink.
///
/// Prefers `canonicalize` (resolves the whole chain, including a relative target, against its real
/// parent directory — a symlink can point through another symlink) and falls back to a single
/// `read_link` hop, manually joined against the link's own parent when the target is relative, for a
/// dangling symlink (a target that doesn't exist yet, which `canonicalize` refuses to resolve at all) —
/// still better than silently destroying the link in that case.
fn resolve_symlink_target(path: &std::path::Path) -> Option<std::path::PathBuf> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if !meta.file_type().is_symlink() {
        return None;
    }
    if let Ok(real) = std::fs::canonicalize(path) {
        return Some(real);
    }
    let target = std::fs::read_link(path).ok()?;
    if target.is_absolute() {
        Some(target)
    } else {
        Some(
            path.parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(target),
        )
    }
}

/// An unpredictable per-call suffix for [`write_atomic`]'s temp file name, so it can't be preplanted
/// under a symlink before the call happens. Same salt-plus-counter construction as
/// `session_store::new_id`: `RandomState` draws OS entropy once per process, and a monotonic counter
/// keeps same-process calls distinct — no need for a `rand`/`uuid` dependency just for this.
fn temp_suffix() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SALT: OnceLock<u64> = OnceLock::new();
    let salt = *SALT.get_or_init(|| RandomState::new().hash_one(0xA70A1Cu64));
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{salt:x}{seq:x}")
}

/// Canonicalize a path for use as a same-file grouping key (`Tool::write_target`): two same-turn calls
/// naming the same underlying file — `./foo.rs`, `foo.rs`, or a symlink alias — must land on one key so
/// the loop serializes them instead of racing two concurrent read-modify-writes on one file.
///
/// Real canonicalization (`std::fs::canonicalize`) resolves symlinks and `.`/`..` against the actual
/// filesystem, but fails with `NotFound` for a path that doesn't exist yet — exactly `write`'s common
/// case (creating a new file). The fallback below only resolves `.`/`..` *lexically* and makes the
/// result absolute (joined against the process cwd), so differently-spelled references to a
/// not-yet-created file still normalize to the same key; it just can't resolve a symlink it can't stat.
///
/// Never fails outright: a pathological input (e.g. one that can't even be joined) falls back to the
/// original string unchanged, which only degrades grouping (calls that should serialize might not) —
/// strictly no worse than today's un-canonicalized behavior, never a new correctness hazard.
///
/// `root` is the tool's configured root (see [`resolve_against`]); a relative `path` is keyed against it
/// rather than against the *process* cwd. This is what makes two subagents running in separate git
/// worktrees, each editing `src/lib.rs`, resolve to two different keys — as they must, since those are
/// two different files. Keying both against one process-wide cwd would serialize them against each other
/// for no reason, and (worse) a shared `WriteLockRegistry` would report them as the same file. An empty
/// `root` means "the process cwd", preserving the pre-root behavior exactly.
pub(crate) fn canonical_write_target(root: &Path, path: &str) -> String {
    if let Ok(real) = std::fs::canonicalize(path) {
        return real.display().to_string();
    }
    lexical_write_target(root, path)
}

/// The purely lexical half of [`canonical_write_target`]: fold away `.` and `..` and join a relative
/// path onto `root`, touching no filesystem at all.
///
/// This is the *entire* body [`canonical_write_target`] already fell back to when `canonicalize`
/// failed — extracted rather than rewritten, so the local behavior is byte-identical.
///
/// It exists as its own function because `canonicalize` is a **host** syscall, and calling it for a
/// path that isn't on this host is worse than useless. With a sandbox mounted at the same absolute
/// path as the working tree — the configuration we actually want, so skills and the prompt's `cwd`
/// stay true — `canonicalize` *succeeds*, resolving the **host's** symlinks, and returns a key for a
/// different file than the one about to be written. Getting the right answer by accident when paths
/// differ and the wrong one in the recommended configuration is the worst possible failure mode, so
/// the choice is made explicitly by [`write_key`] rather than left to whether a syscall happened to
/// fail.
pub(crate) fn lexical_write_target(root: &Path, path: &str) -> String {
    let p = std::path::Path::new(path);
    let mut normalized = if p.is_absolute() {
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
    let resolved = resolve_against_in(root, path, world);
    match world {
        // On this host, `canonicalize` is meaningful and valuable: it unifies `./x`, `../dir/x` and a
        // symlink to `x` into one key, which is what stops a same-turn `edit(real)` + `write(alias)`
        // from racing on one file.
        fs::PathWorld::Local => canonical_write_target(root, &resolved),
        // Elsewhere, it can only lie. See [`lexical_write_target`].
        fs::PathWorld::Remote { .. } => lexical_write_target(root, &resolved),
    }
}

/// Resolve a model-supplied `path` argument against a filesystem tool's configured `root`: normalize it
/// ([`normalize_path`] — `~`, `file://`, a leading `@`, folded Unicode spaces), then, if it is still
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
    resolve_against_in(root, path, &fs::PathWorld::Local)
}

/// [`resolve_against`], but expanding `~` against the home of whichever filesystem `world` names —
/// this host's `$HOME` for [`fs::PathWorld::Local`], the target's for a remote one. The join itself is
/// pure string work and identical either way; only tilde expansion is world-sensitive.
pub(crate) fn resolve_against_in(root: &Path, path: &str, world: &fs::PathWorld) -> String {
    let normalized = match world {
        fs::PathWorld::Local => normalize_path(path),
        fs::PathWorld::Remote { home } => normalize_path_with_home(path, home.as_deref()),
    };
    if root.as_os_str().is_empty() || Path::new(&normalized).is_absolute() {
        return normalized;
    }
    root.join(&normalized).display().to_string()
}

/// Reject any resolved path that isn't a regular file or a directory, *before* a caller ever opens it —
/// shared by `read` (image-sniffing/content), `write` (the overwrite-open path, via [`is_writable`]),
/// and `edit` (`read_with_mtime`'s initial open). Originally lived only in `read.rs`; hoisted here once
/// the same hang class was found in `write`/`edit` too (pi-parity pass 20) — each of those blocks on a
/// syscall no timeout/kill-on-drop can preempt, exactly like `read`'s own pre-fix bug below, so one
/// shared gate closes all three rather than three independently-maintained copies.
///
/// That gap is a real hang, not just a theoretical one: opening a FIFO for reading blocks until a
/// writer shows up on the other end (a bare `mkfifo` with no writer wedges the very first read), opening
/// one for *writing* (`write`'s overwrite path, `OpenOptions::write(true)`) equally blocks until a
/// *reader* shows up, and a character device like `/dev/zero` opens and reads immediately but never
/// signals EOF or contains a newline byte — a line-scanning read loop spins forever waiting for either.
/// None of this has a timeout or kill-on-drop the way `bash`'s child processes do, so any one of them
/// wedges the whole tool call — and, since dispatch awaits each call in turn, the entire turn — forever;
/// worse, because it's a blocking syscall inline in the future's own `poll()` rather than behind an
/// `await` point, dropping/cancelling the enclosing future can't recover it either — only killing the
/// whole process can, which in `serve` pins a tokio worker thread shared by other concurrent sessions.
///
/// There's no legitimate use for reading/writing a FIFO/socket/device through these tools (a named
/// pipe's "content" isn't a stable, replayable value the way a regular file's bytes are, and images are
/// never any of these), so this simply refuses all of them with one clear error rather than trying to
/// special-case any as safe. A directory is let through unmolested — opening one for reading or writing
/// already fails cleanly a moment later (an `EISDIR`-style OS error) — so it needs no separate gate here.
///
/// Uses `std::fs::metadata` (follows symlinks) rather than `symlink_metadata`, so a symlink pointing
/// *at* a FIFO/device is caught too, not just a bare one. A path whose metadata can't be read at all
/// (most commonly: doesn't exist) is let through here — that already has its own clear "missing
/// file"/"not writable" error at the real open call below, and duplicating it here under a different
/// wording would just be a second, inconsistent failure mode for the same cause.
///
/// `tool` names the caller in the error message (`"read"`/`"write"`/`"edit"`) so the model sees which
/// tool actually refused the path, not a generic one-size-fits-all wording.
/// The same refusal, phrased identically, for a caller that already has the path's
/// [`FileKind`](fs::FileKind) from a backend `stat` and must not pay a second, host-only
/// `std::fs::metadata` to rediscover it — which for a remote path would be answering the question
/// about the wrong machine.
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

/// Everything [`default_registry_with_config`] needs to build the tool set.
///
/// This replaces what used to be a ladder of five `default_registry_with_prefix_and_…` functions, each
/// adding one positional knob and delegating down. That convention stopped paying for itself at
/// `default_registry_with_prefix_image_auto_resize_and_mcp_tools`: a sixth knob (`root`, which subagents
/// need) would have meant a sixth name and a sixth wrapper, for two real production call sites
/// (`main.rs::run_task` and `serve.rs::build_tools`). `Default` + struct-update syntax gives every
/// caller the same "only name what you override" ergonomics the ladder existed to provide.
#[derive(Default)]
pub struct ToolConfig<'a> {
    /// `bash`'s default timeout, applied when the model omits `timeout_ms` (`--bash-timeout-ms`).
    pub bash_timeout_ms: Option<u64>,
    /// Which shell `bash` runs commands through (`--bash-shell-path`). Trusted as already validated at
    /// CLI-argument time rather than re-checked on every registry rebuild (`set_model` rebuilds).
    pub bash_shell_path: Option<&'a str>,
    /// A line prepended to every `bash` command in the same shell invocation
    /// (`--bash-command-prefix`, matching pi's `shellCommandPrefix`).
    pub bash_command_prefix: Option<&'a str>,
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

/// The constructed tool set: Pi's seven coding tools (read, write, edit, bash, ls, grep, find), the
/// `todo` task list, and `web`. Advertised default is those eight minus `web`; `web` stays registered
/// so `--tools web` can opt it in. No vendor-specific platform tools, no root (process cwd).
pub fn default_registry() -> ToolRegistry {
    default_registry_with_config(&ToolConfig::new())
}

/// Like [`default_registry`], overriding `bash`'s default timeout and/or shell — the two knobs enough
/// callers (tests, `serve_tools_bash.rs`) want on their own to be worth a shorthand over building a
/// whole [`ToolConfig`].
pub fn default_registry_with(
    bash_timeout_ms: Option<u64>,
    bash_shell_path: Option<&str>,
) -> ToolRegistry {
    default_registry_with_config(&ToolConfig {
        bash_timeout_ms,
        bash_shell_path,
        ..ToolConfig::new()
    })
}

/// Assemble the tool set described by `cfg`. The one real constructor; everything else here is a
/// shorthand over it.
pub fn default_registry_with_config(cfg: &ToolConfig<'_>) -> ToolRegistry {
    let root = cfg.root.as_path();
    let mut reg = ToolRegistry::new();
    // One backend, shared by all six filesystem tools — so they cannot disagree about which
    // filesystem they are acting on, and a `read` can always see what a `write` just wrote.
    let mut read_tool = read::Read::new(root).with_image_auto_resize(cfg.image_auto_resize);
    let mut write_tool = write::Write::new(root);
    let mut edit_tool = edit::Edit::new(root);
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
    let mut bash = match (&cfg.command_runner, cfg.bash_timeout_ms) {
        (Some(runner), Some(ms)) => {
            bash::Bash::with_runner(runner.clone()).with_default_timeout_ms(ms)
        }
        (Some(runner), None) => bash::Bash::with_runner(runner.clone()),
        (None, Some(ms)) => bash::Bash::real().with_default_timeout_ms(ms),
        (None, None) => bash::Bash::real(),
    };
    bash = bash.with_root(root);
    if let Some(path) = cfg.bash_shell_path {
        bash = bash.with_shell_path(path);
    }
    if let Some(prefix) = cfg.bash_command_prefix {
        bash = bash.with_command_prefix(prefix);
    }
    reg.register(Arc::new(bash));
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
    #[cfg(feature = "code-mode")]
    if cfg.code_mode {
        let nested =
            code_mode::select_deferred_tools(cfg.mcp_tools, cfg.nested_exclude, cfg.nested_deny);
        reg.register(Arc::new(code_mode::Execute::new(nested)));
    } else {
        for tool in cfg.mcp_tools {
            reg.register(tool.clone());
        }
    }
    #[cfg(not(feature = "code-mode"))]
    {
        let _ = cfg.code_mode;
        for tool in cfg.mcp_tools {
            reg.register(tool.clone());
        }
    }
    reg
}

/// Restrict a registry to an allow-list, a deny-list, or nothing at all — the CLI/RPC surface for
/// scoping an agent's capabilities (e.g. a read-only reviewer with no `bash`/`edit`/`write`), which
/// otherwise has no way to run with less than the full default tool set. `no_tools` wins outright
/// (checked first); otherwise `tools` (if given) narrows to exactly those names, and `exclude` (if
/// given) drops names from whatever remains — so `tools` + `exclude` can combine to carve one tool out
/// of an allow-list, though picking just one of the three is the common case. Unknown names in either
/// list are silently ignored (there's nothing to remove/keep that isn't already there), matching
/// `ToolRegistry::retain`'s set semantics rather than erroring on a typo.
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
    if no_tools {
        reg.retain(|_| false);
        return;
    }
    if let Some(allow) = tools {
        reg.retain(|name| allow.iter().any(|a| a == name));
    }
    if let Some(deny) = exclude {
        reg.retain(|name| !deny.iter().any(|d| d == name));
    }
    // No `--tools` allow-list: drop opt-in built-ins (`web`). `--tools web` is how it comes back;
    // `--exclude-tools` alone must not re-advertise it.
    if tools.is_none() && !no_tools {
        reg.retain(|name| !OPTIONAL_BUILTINS.iter().any(|t| *t == name));
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::tools::fs::FsBackend as _;

    use super::*;

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
        let dir = tempfile::tempdir().unwrap();
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
    fn default_registry_has_the_coding_tools_and_nothing_vendor_specific() {
        let reg = default_registry();
        // pi's coding tools …
        for name in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert!(reg.get(name).is_some(), "missing coding tool: {name}");
        }
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
        assert_eq!(reg.len(), 9);
    }

    #[test]
    fn advertised_default_omits_web() {
        let mut reg = default_registry();
        apply_filter(&mut reg, None, None, false);
        for name in [
            "read", "write", "edit", "bash", "ls", "grep", "find", "todo",
        ] {
            assert!(reg.get(name).is_some(), "missing advertised tool: {name}");
        }
        assert!(
            reg.get("web").is_none(),
            "web must be opt-in via --tools, not advertised by default"
        );
        assert_eq!(reg.len(), 8);
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
        // Unknown deny names are ignored; the advertised default (eight tools, no web) still applies.
        assert_eq!(reg.len(), 8);
        assert!(reg.get("read").is_some());
        assert!(reg.get("grep").is_some());
        assert!(reg.get("web").is_none());
    }

    #[test]
    fn default_registry_with_none_matches_default_registry() {
        // `default_registry()` must genuinely delegate to `default_registry_with(None, None)`, not
        // just happen to look similar — same tool set either way.
        assert_eq!(
            default_registry().len(),
            default_registry_with(None, None).len()
        );
        for name in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert!(default_registry_with(None, None).get(name).is_some());
        }
    }

    /// A trivial stand-in for an [`mcp::McpTool`] — that type is crate-private and always backed by a
    /// live connection, so tests exercising `default_registry_with_prefix_image_auto_resize_and_mcp_tools`
    /// (which only cares that *some* `Arc<dyn Tool>` merges in under its own name) use this instead of
    /// standing up a real MCP server.
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
        // Nested exclude drops the tool from execute's catalog too — registry still has execute.
        let reg = default_registry_with_config(&ToolConfig {
            mcp_tools: &mcp_tools,
            code_mode: true,
            nested_exclude: &exclude,
            nested_deny: &deny,
            ..ToolConfig::new()
        });
        assert!(reg.get("execute").is_some());
        assert!(reg.get("mcp__filesystem__read_file").is_none());
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
        code_mode::restore_execute(
            &mut reg,
            Arc::new(code_mode::Execute::new(nested)),
            None,
            false,
        );
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
    /// of the three ordinary cases is pinned here — the coverage `reject_non_regular_file`'s own tests
    /// used to provide before the check moved behind the backend.
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
