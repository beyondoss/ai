//! The agent's coding tools — pi's tool set, ported.
//!
//! Each tool implements [`agent_core::Tool`]; [`default_registry`] assembles the set the agent
//! advertises to the model. The Beyond platform tools (fork/sync/logs) register here too once added.

use std::sync::Arc;

use agent_core::{ToolError, ToolRegistry};

pub mod bash;
pub mod beyond;
pub mod edit;
pub mod exec;
pub mod find;
pub mod grep;
pub mod ls;
pub mod output;
pub mod read;
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
    expand_tilde(&folded, home_dir().as_deref())
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
pub(crate) fn canonical_write_target(path: &str) -> String {
    if let Ok(real) = std::fs::canonicalize(path) {
        return real.display().to_string();
    }
    let p = std::path::Path::new(path);
    let mut normalized = if p.is_absolute() {
        std::path::PathBuf::new()
    } else {
        std::env::current_dir().unwrap_or_default()
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
pub(crate) fn reject_non_regular_file(path: &str, tool: &str) -> Result<(), ToolError> {
    let Ok(meta) = std::fs::metadata(path) else {
        return Ok(());
    };
    let file_type = meta.file_type();
    if file_type.is_file() || file_type.is_dir() {
        return Ok(());
    }
    Err(ToolError::Execution(format!(
        "{path} is not a regular file (it's a FIFO, socket, or device); `{tool}` can only operate on \
         regular files"
    )))
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

/// The default tool set: pi's seven coding tools (read, write, edit, bash, ls, grep, find) plus the
/// Beyond platform tools (fork, sync, logs).
pub fn default_registry() -> ToolRegistry {
    default_registry_with(None, None)
}

/// Like [`default_registry`], overriding `bash`'s default timeout (applied when the model omits
/// `timeout_ms`) when `bash_timeout_ms` is `Some`, and/or which shell it runs commands through when
/// `bash_shell_path` is `Some` — operator-tunable knobs (`--bash-timeout-ms`/`--bash-shell-path`),
/// distinct from `default_registry`'s fixed defaults so callers that don't need either override (the
/// `tools` listing command, tests) don't have to pass one. `bash_shell_path` is trusted as already
/// validated (see `--bash-shell-path` in `main.rs`, checked once at CLI-argument time) rather than
/// re-checked on every rebuild this feeds (`set_model`/`set_thinking` each rebuild the registry).
pub fn default_registry_with(
    bash_timeout_ms: Option<u64>,
    bash_shell_path: Option<&str>,
) -> ToolRegistry {
    default_registry_with_prefix(bash_timeout_ms, bash_shell_path, None)
}

/// Like [`default_registry_with`], plus an optional line prepended to every `bash` command in the same
/// shell invocation (`--bash-command-prefix`/`AI_AGENT_BASH_COMMAND_PREFIX`, matching pi's own
/// `shellCommandPrefix` setting) — split out as its own function rather than a third parameter on the
/// widely-called `default_registry_with` so every existing two-argument call site (tests included) stays
/// unchanged. `read`'s image auto-resize defaults on (see
/// [`default_registry_with_prefix_and_image_auto_resize`] for a caller that needs to turn it off).
pub fn default_registry_with_prefix(
    bash_timeout_ms: Option<u64>,
    bash_shell_path: Option<&str>,
    bash_command_prefix: Option<&str>,
) -> ToolRegistry {
    default_registry_with_prefix_and_image_auto_resize(
        bash_timeout_ms,
        bash_shell_path,
        bash_command_prefix,
        true,
    )
}

/// Like [`default_registry_with_prefix`], additionally gating `read`'s image resize/downscale path on
/// `image_auto_resize` (pi's `ImageSettings.autoResize`, `--no-image-auto-resize`/
/// `agent settings --image-auto-resize`) — split out as its own function, rather than a fourth
/// parameter on `default_registry_with_prefix` itself, so that function's existing call sites are
/// unaffected by this addition.
pub fn default_registry_with_prefix_and_image_auto_resize(
    bash_timeout_ms: Option<u64>,
    bash_shell_path: Option<&str>,
    bash_command_prefix: Option<&str>,
    image_auto_resize: bool,
) -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(
        read::Read::default().with_image_auto_resize(image_auto_resize),
    ));
    reg.register(Arc::new(write::Write));
    reg.register(Arc::new(edit::Edit));
    reg.register(Arc::new(ls::Ls));
    reg.register(Arc::new(grep::Grep));
    reg.register(Arc::new(find::Find));
    let mut bash = match bash_timeout_ms {
        Some(ms) => bash::Bash::real().with_default_timeout_ms(ms),
        None => bash::Bash::real(),
    };
    if let Some(path) = bash_shell_path {
        bash = bash.with_shell_path(path);
    }
    if let Some(prefix) = bash_command_prefix {
        bash = bash.with_command_prefix(prefix);
    }
    reg.register(Arc::new(bash));
    reg.register(Arc::new(beyond::Fork::real()));
    reg.register(Arc::new(beyond::Sync::real()));
    reg.register(Arc::new(beyond::Logs::real()));
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
}

#[cfg(test)]
mod tests {
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

        let a = canonical_write_target(real.to_str().unwrap());
        let b = canonical_write_target(dotted.to_str().unwrap());
        let c = canonical_write_target(via_parent.to_str().unwrap());
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

        let via_real = canonical_write_target(real.to_str().unwrap());
        let via_symlink = canonical_write_target(link.to_str().unwrap());
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

        let a = canonical_write_target(base.join("brand-new.rs").to_str().unwrap());
        let b = canonical_write_target(base.join("./brand-new.rs").to_str().unwrap());
        let c = canonical_write_target(base.join("sub/../brand-new.rs").to_str().unwrap());

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert!(
            std::path::Path::new(&a).is_absolute(),
            "fallback must still produce an absolute key: {a}"
        );
    }

    #[test]
    fn default_registry_has_coding_and_beyond_tools() {
        let reg = default_registry();
        // pi's coding tools …
        for name in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert!(reg.get(name).is_some(), "missing coding tool: {name}");
        }
        // … plus the Beyond platform tools.
        for name in ["fork", "sync", "logs"] {
            assert!(reg.get(name).is_some(), "missing beyond tool: {name}");
        }
        assert_eq!(reg.len(), 10);
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
        assert_eq!(reg.len(), default_registry().len() - 3);
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
        let before = reg.len();
        apply_filter(&mut reg, None, Some(&["does-not-exist".to_string()]), false);
        assert_eq!(reg.len(), before);
    }

    #[test]
    fn default_registry_with_none_matches_default_registry() {
        // `default_registry()` must genuinely delegate to `default_registry_with(None, None)`, not
        // just happen to look similar — same tool set either way.
        assert_eq!(
            default_registry().len(),
            default_registry_with(None, None).len()
        );
        for name in [
            "read", "write", "edit", "bash", "ls", "grep", "find", "fork", "sync", "logs",
        ] {
            assert!(default_registry_with(None, None).get(name).is_some());
        }
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
    fn reject_non_regular_file_names_the_calling_tool_in_its_error() {
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

        let err = reject_non_regular_file(fifo_path.to_str().unwrap(), "edit").unwrap_err();
        match err {
            ToolError::Execution(msg) => {
                assert!(msg.contains("`edit`"), "got: {msg}");
                assert!(msg.contains("FIFO"), "got: {msg}");
            }
            other => panic!("expected Execution, got {other:?}"),
        }
    }

    #[test]
    fn reject_non_regular_file_allows_a_regular_file_through() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "x").unwrap();
        assert!(reject_non_regular_file(path.to_str().unwrap(), "write").is_ok());
    }

    #[test]
    fn reject_non_regular_file_allows_a_missing_path_through() {
        // A missing path is left to the caller's own real open call to produce a clearer, tool-specific
        // "not found"/"not writable" error rather than duplicating that here under different wording.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.txt");
        assert!(reject_non_regular_file(path.to_str().unwrap(), "write").is_ok());
    }

    #[test]
    fn reject_non_regular_file_allows_a_directory_through() {
        let dir = tempfile::tempdir().unwrap();
        assert!(reject_non_regular_file(dir.path().to_str().unwrap(), "write").is_ok());
    }
}
