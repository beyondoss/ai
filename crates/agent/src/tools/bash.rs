//! `bash` — run a shell command via a resolved `bash` (falling back to `sh`) and stream/return its
//! combined output.
//!
//! Output handling mirrors pi: raw stdout+stderr feed one [`OutputAccumulator`] (tail-truncated for
//! display, full stream spilled to a temp file → `Full output: <path>`), and while the command runs
//! the tool emits an initial empty update then **throttled snapshots** (the whole output so far, every
//! [`UPDATE_THROTTLE`]) via its [`ToolProgress`] sink. Non-zero exit / timeout become errors carrying
//! the output — same as pi's throw.

use std::borrow::Cow;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput, ToolProgress};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

use super::exec::{ChunkSink, CommandRunner, RealRunner};
use super::output::{OutputAccumulator, OutputSnapshot, TruncatedBy, format_output};

/// Default command timeout (ms) when the model doesn't specify one. The reference agent has no default
/// at all — a command runs to completion unless the model explicitly sets `timeout_ms` — which SIGKILLs
/// long builds/tests the model didn't think to extend. We deliberately deviate: this agent runs
/// unattended on a homelab node with no one watching a hung shell, so a runaway/blocked command needs a
/// backstop. 30 minutes is generous enough to never bite a real build/test/install, while still
/// bounding a truly stuck command instead of leaving it to hang a turn forever.
const DEFAULT_TIMEOUT_MS: u64 = 1_800_000;
/// Ceiling on a model-supplied `timeout_ms` — matches pi's own `MAX_TIMEOUT_MS` (`i32::MAX`
/// milliseconds, ~24.8 days; Node's `setTimeout` silently misbehaves past a 32-bit delay, the original
/// reason for the exact number, but kept here too for parity and because the sanity bound it enforces
/// — reject an absurd value instead of quietly running an effectively-unbounded command — applies
/// regardless of platform). Without this, an accidental or pathological huge value (a model typo, or a
/// deliberately adversarial one) would defeat the whole point of having a timeout at all.
const MAX_TIMEOUT_MS: u64 = 2_147_483_647;
/// Minimum gap between streamed progress snapshots — pi's `BASH_UPDATE_THROTTLE_MS`. Keeps a chatty
/// command from flooding the event stream; the final snapshot is always emitted regardless.
const UPDATE_THROTTLE: Duration = Duration::from_millis(100);

/// Resolve the shell commands run through: `/bin/bash` if present, else `bash` on `$PATH`, else `sh` —
/// pi's `shell.ts` resolution order (minus the Windows/WSL branches, which don't apply on this
/// platform). Bash's associative arrays, `[[`, `pipefail`, and process substitution are common enough
/// in model-generated commands that silently falling back to a POSIX `sh` (which may be `dash`, and
/// rejects all of the above) is a real correctness gap, not just a style preference. Cached: the
/// filesystem/PATH answer can't change mid-process.
fn resolve_shell() -> &'static str {
    static SHELL: OnceLock<String> = OnceLock::new();
    SHELL.get_or_init(|| {
        if Path::new("/bin/bash").exists() {
            return "/bin/bash".to_string();
        }
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                let candidate = dir.join("bash");
                if candidate.exists() {
                    return candidate.to_string_lossy().into_owned();
                }
            }
        }
        "sh".to_string()
    })
}

pub struct Bash {
    runner: Arc<dyn CommandRunner>,
    default_timeout_ms: u64,
    /// Overrides `resolve_shell()`'s auto-detection when set — see [`with_shell_path`](Self::with_shell_path).
    shell_path: Option<String>,
    /// Prepended to every command, on its own line, when set — see
    /// [`with_command_prefix`](Self::with_command_prefix).
    command_prefix: Option<String>,
    /// Rendered once (at construction, and again if `with_default_timeout_ms` changes the default) —
    /// see [`describe`] — rather than a `'static` literal, so a customized default actually shows up
    /// in what the model sees instead of the tool description silently going stale for a deployment
    /// that overrides it via `--bash-timeout-ms`.
    description: String,
}

/// Build the model-facing tool description, stating the *actual* default timeout (a model omitting
/// `timeout_ms` has no other way to know a command will be killed after this long, unlike the
/// truncation budget below, which is at least self-discoverable via the `"Full output: <path>"` marker
/// once it actually happens) and the output truncation budget (matching pi's own bash tool description,
/// which documents both) — the model shouldn't have to learn either from a failed run.
fn describe(default_timeout_ms: u64) -> String {
    format!(
        "Run a shell command via a resolved bash (falls back to sh) and return its combined \
         stdout/stderr. Supports an optional `cwd` and `timeout_ms` (defaults to {default_timeout_ms} ms \
         / {} minutes if omitted). Output is truncated to the last {} lines or {}, whichever is hit \
         first; if truncated, the complete output is saved to a temp file you can read.",
        default_timeout_ms / 60_000,
        super::output::DEFAULT_MAX_LINES,
        super::output::format_size(super::output::DEFAULT_MAX_BYTES as u64),
    )
}

impl Bash {
    /// A `bash` tool that runs commands for real.
    pub fn real() -> Self {
        Self {
            runner: Arc::new(RealRunner),
            default_timeout_ms: DEFAULT_TIMEOUT_MS,
            shell_path: None,
            command_prefix: None,
            description: describe(DEFAULT_TIMEOUT_MS),
        }
    }

    /// Builder-style: override the default timeout applied when the model omits `timeout_ms`.
    pub fn with_default_timeout_ms(mut self, ms: u64) -> Self {
        self.default_timeout_ms = ms;
        self.description = describe(ms);
        self
    }

    /// Builder-style: run commands through this shell instead of the auto-resolved one
    /// (`resolve_shell()`: `/bin/bash`, else `bash` on `$PATH`, else `sh`) — for a non-standard
    /// environment (Cygwin, a container without `/bin/bash` at the expected path, a hardened/audited
    /// shell wrapper) where auto-detection would pick the wrong binary. Matches pi's own `shellPath`
    /// setting (`getShellConfig(customShellPath)`). Existence is the caller's responsibility to check
    /// up front (see `--bash-shell-path` in `main.rs`) — failing fast at CLI-argument time, once, is
    /// simpler than threading a `Result` through every tool-registry rebuild this would otherwise
    /// touch (`set_model`/`set_thinking` each rebuild the registry from `ServeConfig`).
    pub fn with_shell_path(mut self, path: impl Into<String>) -> Self {
        self.shell_path = Some(path.into());
        self
    }

    /// Builder-style: prepend `prefix` (its own line, before the model's command) to every command run
    /// through this tool — matches pi's `BashToolOptions.commandPrefix` (e.g. sourcing a project's env
    /// setup, activating a venv, or setting a variable every command should see). Both the prefix's and
    /// the command's output land in the same combined stdout/stderr stream, in order — this is just
    /// script concatenation, not a separate execution.
    pub fn with_command_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.command_prefix = Some(prefix.into());
        self
    }

    /// A `bash` tool over a custom runner (tests inject one to capture the invocation).
    #[cfg(test)]
    pub fn with_runner(runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            runner,
            default_timeout_ms: DEFAULT_TIMEOUT_MS,
            shell_path: None,
            command_prefix: None,
            description: describe(DEFAULT_TIMEOUT_MS),
        }
    }

    /// The shell `exec()` invokes: the override from [`with_shell_path`](Self::with_shell_path) if
    /// set, else the auto-resolved default.
    fn shell(&self) -> &str {
        self.shell_path
            .as_deref()
            .unwrap_or_else(|| resolve_shell())
    }

    /// Shared body for [`Tool::run`] and [`Tool::run_streaming`]. Raw output feeds one accumulator; when
    /// `progress` is set, snapshots stream live. The returned value is the cleaned, tail-truncated
    /// output with pi's `Full output:` marker; a non-zero exit or timeout is an error carrying it.
    async fn exec(
        &self,
        input: Value,
        progress: Option<&ToolProgress>,
    ) -> Result<ToolOutput, ToolError> {
        let command = input
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `command`".into()))?;
        // Pi-parity note: `cwd` being a per-call, model-settable field here is a deliberate DIVERGENCE,
        // not a gap to close. Pi's own bash tool schema has no `cwd` at all — its working directory is
        // fixed once at construction (the shell config baked in at startup), not choosable per
        // invocation. Letting the model pick `cwd` per call is a real capability beyond's own, kept as
        // a plausible enhancement rather than narrowed to match.
        let cwd = input.get("cwd").and_then(Value::as_str);
        // Fail with a clear message instead of the raw spawn-error wrapping ("spawn failed: No such
        // file or directory") a bad `cwd` would otherwise surface as — matches pi's own pre-check, plus
        // a stricter, more accurate rejection: pi only checks existence, so it would happily hand a
        // *file* path to the shell as `cwd` and let the spawn itself fail with a confusing "Not a
        // directory" OS error. `.is_dir()` rejects that up front too — but that means "does not exist"
        // and "exists but isn't a directory" must be told apart explicitly (pi-parity task 51): a path
        // that exists (a file, a broken permission on an ancestor notwithstanding) must not be reported
        // as nonexistent.
        if let Some(dir) = cwd {
            let p = Path::new(dir);
            if !p.is_dir() {
                let msg = if p.exists() {
                    format!("Working directory is not a directory: {dir}")
                } else {
                    // pi-parity fix: pi's own bash.ts message is two lines — the plain fact, plus this
                    // explanatory second line stating the consequence — not just the first on its own.
                    format!(
                        "Working directory does not exist: {dir}\nCannot execute bash commands."
                    )
                };
                return Err(ToolError::InvalidInput(msg));
            }
        }
        // A present `timeout_ms` must be a positive integer no larger than `MAX_TIMEOUT_MS` — 0 would
        // pass straight through as an instant, confusingly-worded timeout ("Command timed out after 0
        // seconds"), a negative value previously fell silently back to the default instead of being
        // rejected as the obvious caller mistake it is, and an absurdly large one would defeat the
        // whole point of having a timeout at all (matching pi's `resolveTimeoutMs` validation, upper
        // bound included). A missing key still means "use the default", not an error.
        let timeout_ms = match input.get("timeout_ms") {
            None | Some(Value::Null) => self.default_timeout_ms,
            Some(v) => match v.as_u64() {
                Some(ms) if ms > 0 && ms <= MAX_TIMEOUT_MS => ms,
                Some(ms) if ms > MAX_TIMEOUT_MS => {
                    return Err(ToolError::InvalidInput(format!(
                        "`timeout_ms` must be at most {MAX_TIMEOUT_MS}, got {ms}"
                    )));
                }
                _ => {
                    return Err(ToolError::InvalidInput(format!(
                        "`timeout_ms` must be a positive integer, got {v}"
                    )));
                }
            },
        };
        // Prefix and command run as one script in one shell invocation — matches pi's
        // `${commandPrefix}\n${command}` — so both land in the same combined output stream, in order,
        // rather than looking like two separate calls.
        let resolved_command = match &self.command_prefix {
            Some(prefix) => format!("{prefix}\n{command}"),
            None => command.to_string(),
        };
        let args = vec!["-c".to_string(), resolved_command];
        let dur = Duration::from_millis(timeout_ms);

        let acc = Arc::new(Mutex::new(OutputAccumulator::new()));
        let streamed = Arc::new(AtomicBool::new(false));
        let last_emit = Arc::new(Mutex::new(Instant::now()));

        // pi emits an initial empty update the moment execution starts.
        if let Some(p) = progress {
            p.emit("", None);
        }

        // Feed every raw chunk (both stdout and stderr, in arrival order) into the one accumulator, and
        // emit a throttled snapshot as it grows. Always via the streaming path so the temp-file spill
        // sees the *complete* output; a non-streaming runner (test double) delivers no chunks and is
        // handled by the fallback below.
        //
        // `sink` is declared at this scope (not nested in its own block) because `run_fut` borrows it
        // and, unlike before this fix, is no longer awaited to completion in the same expression that
        // constructs it — a cancellation races it instead (see below), so `sink` must outlive that race.
        let sink = {
            let acc = acc.clone();
            let streamed = streamed.clone();
            let last_emit = last_emit.clone();
            move |bytes: &[u8]| {
                streamed.store(true, Ordering::Relaxed);
                let mut a = lock(&acc);
                a.append(bytes);
                if let Some(p) = progress {
                    let mut le = lock(&last_emit);
                    if le.elapsed() >= UPDATE_THROTTLE {
                        *le = Instant::now();
                        drop(le);
                        // pi-parity fix (task 58): pi forces a spill to the temp file on *every* live
                        // progress emit once the output is truncated (50KB), not just on the final
                        // snapshot — `false` here left a live update in the 50-100KB range (`truncated`
                        // already true, but under the ~100KB rolling cap that auto-spills during
                        // `append`) reporting `truncated: true` with no `full_output_path` to point at
                        // yet. `snapshot`'s own `persist_if_truncated` gate means this is a no-op unless
                        // the 50KB mark has actually been crossed, and a no-op again once already
                        // spilled, so this doesn't force an extra file open on every throttled tick.
                        let snap = a.snapshot(true);
                        emit_update(p, &snap);
                    }
                }
            }
        };
        let sink: ChunkSink<'_> = &sink;
        let run_fut = self
            .runner
            .run_streaming(self.shell(), &args, cwd, dur, sink);

        // Races the runner against this call's own cancellation — the same token a caller trips via
        // `abort_bash`/SIGTERM/etc (e.g. `serve`'s host `bash` RPC) — rather than being cancellable only
        // by an external `Drop` of this whole future with no chance to finalize. `acc` is always current
        // regardless of the throttled snapshots above (every chunk is appended unconditionally, matching
        // pi's own accumulator), so on cancellation we flush exactly what's in it and return that, tagged
        // "Command cancelled", instead of discarding it. Without `progress` there's no token to race
        // against, so this just awaits the runner directly, unchanged from before.
        let result = match progress {
            Some(p) => {
                tokio::pin!(run_fut);
                tokio::select! {
                    r = &mut run_fut => Some(r),
                    () = p.cancelled() => None,
                }
            }
            None => Some(run_fut.await),
        };
        let Some(result) = result
            .transpose()
            .map_err(|e: std::io::Error| ToolError::Execution(format!("spawn failed: {e}")))?
        else {
            // Cancelled mid-stream: dropping the pinned `run_fut` above (implicit at the end of the
            // `Some(p)` match arm) kills the subprocess via its `kill_on_drop`/process-group guard, the
            // same way an external `Drop` of this whole call already did before this fix — the only
            // change is that we get a chance to flush `acc` first instead of losing it.
            let snap = {
                let mut a = lock(&acc);
                a.finish();
                a.snapshot(true)
            };
            if let Some(p) = progress {
                emit_update(p, &snap);
            }
            let text = clean(format_output(&snap, ""));
            return Err(ToolError::Execution(append_status(
                &text,
                "Command cancelled",
            )));
        };

        // Fallback for a non-streaming runner (test double): feed its final captured output.
        if !streamed.load(Ordering::Relaxed) {
            let mut a = lock(&acc);
            a.append(result.stdout.as_bytes());
            if !result.stderr.is_empty() {
                if !result.stdout.is_empty() {
                    a.append(b"\n");
                }
                a.append(result.stderr.as_bytes());
            }
        }

        let snap = {
            let mut a = lock(&acc);
            a.finish();
            a.snapshot(true) // persist the full output to the temp file if truncated
        };

        // pi flushes a final snapshot before returning the result.
        if let Some(p) = progress {
            emit_update(p, &snap);
        }

        if result.timed_out {
            let secs = timeout_ms / 1000;
            // No "(no output)" placeholder here: an empty-output timeout should read as just the
            // status line ("Command timed out after Ns"), not "(no output)" glued in front of it —
            // matching the reference agent's abort/timeout formatting, which substitutes nothing for
            // empty content on this path specifically (the success/exit-code path below keeps the
            // placeholder, since there the empty case is a genuinely silent command, not an interrupt).
            let text = clean(format_output(&snap, ""));
            return Err(ToolError::Execution(append_status(
                &text,
                &format!("Command timed out after {secs} seconds"),
            )));
        }
        let text = clean(format_output(&snap, "(no output)"));
        match result.code {
            Some(0) | None => Ok(text.into()),
            // Non-zero exit is an error result that still carries the output — pi throws here too.
            Some(code) => Err(ToolError::Execution(append_status(
                &text,
                &format!("Command exited with code {code}"),
            ))),
        }
    }
}

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to run." },
                "cwd": { "type": "string", "description": "Working directory." },
                "timeout_ms": { "type": "integer", "description": "Timeout in milliseconds." }
            },
            "required": ["command"]
        })
    }

    // An opaque shell command can mutate anything reachable from its `cwd` — there's no single path
    // to report via `write_target`, so without this a same-turn `edit`/`write` it happens to race
    // (e.g. `edit foo.py` batched with `bash: black foo.py`) would run fully concurrently against it.
    fn conservative_exclusive(&self) -> bool {
        true
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        self.exec(input, None).await
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        self.exec(input, Some(progress)).await
    }
}

/// Lock a mutex, recovering the data on poison rather than panicking (the crate forbids `unwrap`).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Emit one streamed progress snapshot: the cleaned output so far, plus truncation + full-output-path
/// details when the output has overflowed (pi's `onUpdate({content, details})`).
///
/// Uses [`clean_str`] (borrowing) rather than [`clean`] (owning): `snap` only ever arrives by reference
/// here, so `clean` would need a `.clone()` of the whole accumulated content on every throttled tick just
/// to have something to hand it — `clean_str` reads `&snap.content` directly and only allocates when the
/// content actually needs stripping, same as `clean` already did internally.
fn emit_update(p: &ToolProgress, snap: &OutputSnapshot) {
    p.emit(clean_str(&snap.content), truncation_details(snap));
}

/// The `details` payload for a progress update: nested `{truncation: {...}, full_output_path}` —
/// matching the reference agent's `{truncation, fullOutputPath}` shape structurally (field names stay
/// snake_case: this is the only place in the whole wire protocol that would otherwise mix camelCase
/// into an all-snake_case convention, and nothing here needs byte-for-byte pi compatibility, just the
/// same information). The full `Truncation` record — not just the four fields this used to send — so a
/// future client doesn't have to guess `truncated_by`/`last_line_partial`/`max_lines`/`max_bytes` from
/// the others. `None` when the output was never truncated (nothing extra to report).
fn truncation_details(snap: &OutputSnapshot) -> Option<Value> {
    snap.truncation.truncated.then(|| {
        let by = match snap.truncation.truncated_by {
            Some(TruncatedBy::Lines) => Some("lines"),
            Some(TruncatedBy::Bytes) => Some("bytes"),
            None => None,
        };
        json!({
            "truncation": {
                "truncated": true,
                "truncated_by": by,
                "total_lines": snap.truncation.total_lines,
                "total_bytes": snap.truncation.total_bytes,
                "output_lines": snap.truncation.output_lines,
                "output_bytes": snap.truncation.output_bytes,
                "last_line_partial": snap.truncation.last_line_partial,
                "max_lines": snap.truncation.max_lines,
                "max_bytes": snap.truncation.max_bytes,
            },
            "full_output_path": snap.full_output_path,
        })
    })
}

/// Append a status line (`exit code` / `timed out` / `cancelled`) after the output, pi-style
/// (`text\n\n<status>`).
fn append_status(text: &str, status: &str) -> String {
    if text.is_empty() {
        status.to_string()
    } else {
        format!("{text}\n\n{status}")
    }
}

/// ANSI escape + OSC sequences a terminal emits but the model can't use: CSI/SGR colour and cursor
/// moves (`ESC [ … final`, or the 8-bit C1 CSI introducer `\x9b`) plus OSC strings (`ESC ] … `,
/// terminated by BEL, ST (`ESC \`), or the 8-bit C1 ST `\x9c`). A direct port of pi's own
/// `ansi.ts::ansiRegex` (itself derived from the `ansi-regex`/`strip-ansi` npm packages — see that
/// file's license header) rather than a from-scratch pattern, so it inherits the exact same coverage:
/// the prior version only recognized a BEL-terminated OSC (missing an OSC-8 hyperlink or any other
/// modern tool that ST-terminates instead) and excluded `:` from CSI parameters (missing 24-bit
/// truecolor SGR, e.g. `\x1b[38:2:255:0:0m`, which uses colon-separated sub-parameters) — both leaked
/// raw escape-sequence noise into the model's context instead of being stripped. Built once; the
/// pattern is a static literal, so a build failure is impossible — we model it as `None` instead of
/// unwrapping.
fn ansi_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?:\x1b\][\s\S]*?(?:\x07|\x1b\x5c|\x9c))|[\x1b\x9b][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]",
        )
        .ok()
    })
    .as_ref()
}

/// Whitespace the model actually needs (`\t`, `\n`, `\r`) survives; every other C0 control byte (NUL,
/// stray ESC, …) and the Unicode interlinear-annotation controls are dropped so binary output can't
/// corrupt the context.
fn is_keepable(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || (c >= ' ' && !('\u{fff9}'..='\u{fffb}').contains(&c))
}

/// Strip terminal control noise from output. Two single passes: a regex removes ANSI escape sequences,
/// then we drop residual control bytes. When the text is already clean we reuse the string we have
/// rather than reallocating. (pi sanitizes binary output too; we also drop ANSI, which the model can't
/// use — a small, deliberate step past pi that only ever removes noise.)
fn clean(s: String) -> String {
    match clean_str(&s) {
        Cow::Borrowed(_) => s,
        Cow::Owned(owned) => owned,
    }
}

/// [`clean`]'s own logic, borrowing: returns `s` itself (via `Cow::Borrowed`) when nothing needed
/// stripping, so a caller that only has `&str` (e.g. [`emit_update`], reading a snapshot it doesn't own)
/// isn't forced to clone the whole input up front just to satisfy an owned-`String` signature.
fn clean_str(s: &str) -> Cow<'_, str> {
    let stripped = match ansi_re() {
        Some(re) => re.replace_all(s, ""),
        None => Cow::Borrowed(s),
    };
    if stripped.chars().all(is_keepable) {
        return stripped;
    }
    Cow::Owned(stripped.chars().filter(|&c| is_keepable(c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::exec::ExecResult;
    use crate::tools::output::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, Truncation, format_size};

    #[test]
    fn truncation_details_nests_the_full_record_under_a_truncation_key() {
        let snap = OutputSnapshot {
            content: "tail".into(),
            truncation: Truncation {
                truncated: true,
                truncated_by: Some(TruncatedBy::Lines),
                total_lines: 2500,
                total_bytes: 12345,
                output_lines: 2000,
                output_bytes: 9000,
                last_line_partial: true,
                max_lines: DEFAULT_MAX_LINES,
                max_bytes: DEFAULT_MAX_BYTES,
            },
            full_output_path: Some("/tmp/pi-bash-abc.log".into()),
            last_line_bytes: 4,
        };
        let details = truncation_details(&snap).expect("truncated snapshot must carry details");
        assert_eq!(details["truncation"]["truncated"], true);
        assert_eq!(details["truncation"]["truncated_by"], "lines");
        assert_eq!(details["truncation"]["total_lines"], 2500);
        assert_eq!(details["truncation"]["total_bytes"], 12345);
        assert_eq!(details["truncation"]["output_lines"], 2000);
        assert_eq!(details["truncation"]["output_bytes"], 9000);
        assert_eq!(details["truncation"]["last_line_partial"], true);
        assert_eq!(details["truncation"]["max_lines"], DEFAULT_MAX_LINES);
        assert_eq!(details["truncation"]["max_bytes"], DEFAULT_MAX_BYTES);
        assert_eq!(details["full_output_path"], "/tmp/pi-bash-abc.log");
        // `full_output_path` is a sibling of `truncation`, not nested inside it.
        assert!(details["truncation"].get("full_output_path").is_none());
    }

    #[test]
    fn truncation_details_is_none_when_output_was_not_truncated() {
        let snap = OutputSnapshot {
            content: "all of it".into(),
            truncation: Truncation {
                truncated: false,
                truncated_by: None,
                total_lines: 3,
                total_bytes: 9,
                output_lines: 3,
                output_bytes: 9,
                last_line_partial: false,
                max_lines: DEFAULT_MAX_LINES,
                max_bytes: DEFAULT_MAX_BYTES,
            },
            full_output_path: None,
            last_line_bytes: 3,
        };
        assert!(truncation_details(&snap).is_none());
    }

    /// Records the last invocation and returns a canned result (a non-streaming runner: it never
    /// delivers chunks, so `bash` exercises its fallback-from-`ExecResult` path).
    struct RecordingRunner {
        last: std::sync::Mutex<Option<(String, Vec<String>, Duration)>>,
        result: ExecResult,
    }

    #[async_trait]
    impl CommandRunner for RecordingRunner {
        async fn run(
            &self,
            program: &str,
            args: &[String],
            _cwd: Option<&str>,
            timeout: Duration,
        ) -> std::io::Result<ExecResult> {
            *self.last.lock().unwrap() = Some((program.to_string(), args.to_vec(), timeout));
            Ok(self.result.clone())
        }
    }

    fn recording(result: ExecResult) -> Arc<RecordingRunner> {
        Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result,
        })
    }

    /// A streaming runner that replays a fixed sequence of raw byte chunks through `on_chunk`, then
    /// returns a canned final result — for exercising the live-streaming path (`RecordingRunner` above
    /// never streams at all).
    struct ChunkedRunner {
        chunks: Vec<Vec<u8>>,
    }

    #[async_trait]
    impl CommandRunner for ChunkedRunner {
        async fn run(
            &self,
            _program: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _timeout: Duration,
        ) -> std::io::Result<ExecResult> {
            unreachable!("test only exercises the streaming path")
        }

        async fn run_streaming(
            &self,
            _program: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _timeout: Duration,
            on_chunk: ChunkSink<'_>,
        ) -> std::io::Result<ExecResult> {
            for chunk in &self.chunks {
                on_chunk(chunk);
            }
            Ok(ExecResult {
                code: Some(0),
                ..Default::default()
            })
        }
    }

    #[tokio::test]
    async fn invokes_the_resolved_shell_dash_c() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "hi\n".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner.clone())
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap()
            .text;
        // pi does not trim: the command's output is shown as-is (trailing newline kept).
        assert_eq!(out, "hi\n");
        let (prog, args, _timeout) = runner.last.lock().unwrap().clone().unwrap();
        assert_eq!(prog, resolve_shell());
        assert_eq!(args, vec!["-c".to_string(), "echo hi".to_string()]);
    }

    #[tokio::test]
    async fn with_shell_path_overrides_the_auto_resolved_shell() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "hi\n".into(),
            ..Default::default()
        });
        Bash::with_runner(runner.clone())
            .with_shell_path("/bin/dash")
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap();
        let (prog, _, _) = runner.last.lock().unwrap().clone().unwrap();
        assert_eq!(prog, "/bin/dash");
        // The auto-resolution path is unaffected when no override is set.
        assert_ne!(resolve_shell(), "/bin/dash");
    }

    #[test]
    fn resolve_shell_never_panics_and_returns_a_plausible_path() {
        let shell = resolve_shell();
        assert!(!shell.is_empty());
        assert!(shell == "sh" || shell.contains("bash"));
        // Cached: calling twice returns the same answer.
        assert_eq!(resolve_shell(), shell);
    }

    #[tokio::test]
    async fn with_default_timeout_ms_overrides_the_default_when_the_model_omits_one() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "hi\n".into(),
            ..Default::default()
        });
        Bash::with_runner(runner.clone())
            .run(json!({ "command": "echo hi" })) // no `timeout_ms` — the tool's default applies
            .await
            .unwrap();
        let (_, _, timeout) = runner.last.lock().unwrap().clone().unwrap();
        assert_eq!(timeout, Duration::from_millis(DEFAULT_TIMEOUT_MS));

        // `Bash::with_runner` (the test constructor) always uses `DEFAULT_TIMEOUT_MS`; confirm
        // `with_default_timeout_ms` actually changes what gets resolved when explicitly built.
        let runner2 = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        Bash::with_runner(runner2.clone())
            .with_default_timeout_ms(5_000)
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap();
        let (_, _, timeout2) = runner2.last.lock().unwrap().clone().unwrap();
        assert_eq!(timeout2, Duration::from_millis(5_000));
    }

    #[test]
    fn description_documents_the_default_timeout_and_truncation_budget() {
        let desc = Bash::real().description().to_string();
        assert!(
            desc.contains(&DEFAULT_TIMEOUT_MS.to_string()) && desc.contains("30 minutes"),
            "description should state the default timeout, got: {desc}"
        );
        assert!(
            desc.contains(&DEFAULT_MAX_LINES.to_string())
                && desc.contains(&format_size(DEFAULT_MAX_BYTES as u64)),
            "description should state the truncation budget, got: {desc}"
        );

        // A customized default must show up too — a stale literal wouldn't reflect it.
        let custom = Bash::real().with_default_timeout_ms(90_000);
        assert!(
            custom.description().contains("90000") && custom.description().contains("1 minutes"),
            "description should reflect a customized default, got: {}",
            custom.description()
        );
    }

    #[tokio::test]
    async fn zero_or_negative_timeout_ms_is_rejected_not_silently_coerced() {
        let runner = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        let bash = Bash::with_runner(runner.clone());

        // 0 previously passed straight through as an instant, confusing timeout.
        let err = bash
            .run(json!({ "command": "echo hi", "timeout_ms": 0 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "got: {err:?}");

        // A negative value previously fell silently back to the default instead of being rejected.
        let err = bash
            .run(json!({ "command": "echo hi", "timeout_ms": -100 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "got: {err:?}");

        // Neither invalid call should have reached the runner at all.
        assert!(runner.last.lock().unwrap().is_none());

        // A missing `timeout_ms` still means "use the default" — not an error.
        bash.run(json!({ "command": "echo hi" })).await.unwrap();
        assert!(runner.last.lock().unwrap().is_some());
    }

    #[tokio::test]
    async fn a_timeout_ms_past_the_max_is_rejected_not_treated_as_effectively_unbounded() {
        // Matches pi's `resolveTimeoutMs` upper-bound check: an absurdly large value (a typo, or a
        // deliberately adversarial one) would otherwise defeat the entire point of having a timeout —
        // `Duration::from_millis` happily accepts it and the command just never gets killed.
        let runner = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        let bash = Bash::with_runner(runner.clone());

        let err = bash
            .run(json!({ "command": "echo hi", "timeout_ms": MAX_TIMEOUT_MS + 1 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "got: {err:?}");
        assert!(runner.last.lock().unwrap().is_none());

        // The max itself is still accepted (an inclusive bound, not exclusive).
        bash.run(json!({ "command": "echo hi", "timeout_ms": MAX_TIMEOUT_MS }))
            .await
            .unwrap();
        assert!(runner.last.lock().unwrap().is_some());
    }

    #[tokio::test]
    async fn nonzero_exit_is_an_error_carrying_the_output() {
        let runner = recording(ExecResult {
            code: Some(2),
            stderr: "boom".into(),
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "false" }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };
        assert!(msg.contains("boom"));
        assert!(msg.contains("Command exited with code 2"));
    }

    #[tokio::test]
    async fn real_runner_executes() {
        let out = Bash::real()
            .run(json!({ "command": "printf done" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "done");
    }

    #[tokio::test]
    async fn command_prefix_runs_before_the_command_in_the_same_shell() {
        // pi: tools.test.ts, "should prepend command prefix when configured" — the prefix must run in
        // the *same* shell invocation as the command (a variable it sets is visible to the command).
        let out = Bash::real()
            .with_command_prefix("export TEST_VAR=hello")
            .run(json!({ "command": "echo $TEST_VAR" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim(), "hello");
    }

    #[tokio::test]
    async fn command_prefix_and_command_output_both_appear_in_order() {
        // pi: tools.test.ts, "should include output from both prefix and command".
        let out = Bash::real()
            .with_command_prefix("echo prefix-output")
            .run(json!({ "command": "echo command-output" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim(), "prefix-output\ncommand-output");
    }

    #[tokio::test]
    async fn no_command_prefix_runs_the_command_unmodified() {
        // pi: tools.test.ts, "should work without command prefix".
        let out = Bash::real()
            .run(json!({ "command": "echo no-prefix" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim(), "no-prefix");
    }

    #[tokio::test]
    async fn strips_ansi_escape_sequences() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b[31mred\x1b[0m \x1b[2Aup\x1b]0;title\x07 done".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "red up done");
    }

    #[tokio::test]
    async fn strips_an_st_terminated_osc_sequence_not_just_bel_terminated() {
        // Pi-parity audit M9: the prior regex only recognized `ESC ] … BEL` — an OSC-8 hyperlink (or
        // any modern tool) terminated with ST (`ESC \`) instead leaked raw escape-sequence noise into
        // the model's context. Matches pi's own `ansi.ts`, which accepts either terminator.
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b]8;;http://example.com\x1b\\link text\x1b]8;;\x1b\\ done".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "link text done");
    }

    #[tokio::test]
    async fn strips_a_colon_separated_truecolor_sgr_sequence() {
        // Pi-parity audit M9: the prior regex excluded `:` from CSI parameters entirely, so a 24-bit
        // truecolor SGR sequence (`\x1b[38:2:255:0:0m`, colon-separated sub-parameters) wasn't
        // recognized at all and leaked through verbatim. Matches pi's own `ansi.ts`.
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b[38:2:255:0:0mRED\x1b[0m done".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "RED done");
    }

    #[tokio::test]
    async fn sanitizes_control_characters() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "a\x00b\x07c\td\ne".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "abc\td\ne");
    }

    #[tokio::test]
    async fn large_output_keeps_the_tail_and_points_at_the_full_file() {
        // 5000 lines exceed the 2000-line cap: the accumulator keeps the tail, reports the range, and
        // spills the complete output to a temp file the marker names (pi's `Full output:`).
        let body = (0..5000)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: body,
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("[Showing lines"), "range marker present");
        assert!(out.contains("Full output:"), "full-output path present");
        assert!(out.contains("4999"), "tail kept (the last line is present)");
        assert!(
            !out.starts_with("0\n1\n"),
            "head dropped (tail truncation, not head+tail)"
        );
    }

    #[tokio::test]
    async fn timeout_is_an_error() {
        let runner = recording(ExecResult {
            timed_out: true,
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "sleep 10", "timeout_ms": 5000 }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };
        assert!(msg.contains("timed out"));
    }

    #[tokio::test]
    async fn a_timed_out_command_with_truncated_output_still_reports_the_full_output_path() {
        // pi: tools.test.ts, "should include full output path for truncated timeout and abort errors"
        // — iterates the timeout and aborted cases against 3000 lines of output. `truncation` and
        // `timeout` each have their own dedicated test above, but neither proves they compose: a
        // command that was both chatty *and* killed for running too long must still carry the
        // `[Showing lines …]`/`Full output: <path>` markers on its timeout error, with the complete
        // (untruncated) output actually readable from that path — not just an empty "timed out" message
        // that quietly drops everything the command had already printed.
        let body = (1..=3000)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let runner = recording(ExecResult {
            stdout: body.clone(),
            timed_out: true,
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "chatty-and-slow", "timeout_ms": 5000 }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };

        assert!(
            msg.contains("Command timed out after 5 seconds"),
            "got: {msg}"
        );
        assert!(
            msg.contains("[Showing lines"),
            "range marker missing: {msg}"
        );
        assert!(msg.contains("2999"), "tail must be kept: {msg}");
        assert!(
            !msg.contains("1\n2\n3\n4"),
            "head must be dropped (tail truncation, not head+tail): {msg}"
        );

        let path = msg
            .split("Full output: ")
            .nth(1)
            .and_then(|s| s.split(']').next())
            .expect("must carry a full-output path");
        assert!(
            std::path::Path::new(path).exists(),
            "full-output file must actually exist: {path}"
        );
        let full = std::fs::read_to_string(path).unwrap();
        assert_eq!(
            full, body,
            "the complete, untruncated output must be recoverable from the full-output file even \
             though the command timed out"
        );
    }

    #[tokio::test]
    async fn nonexistent_cwd_is_a_clear_invalid_input_error_not_a_raw_spawn_failure() {
        let runner = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        let err = Bash::with_runner(runner.clone())
            .run(json!({ "command": "echo hi", "cwd": "/definitely/not/a/real/path/xyz" }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => {
                assert!(
                    msg.contains("Working directory does not exist"),
                    "got: {msg}"
                );
                // pi-parity fix: pi's own bash.ts message is two lines — the plain fact, plus this
                // explanatory second line stating the consequence ("Cannot execute bash commands.") —
                // not just the first line on its own.
                assert!(msg.contains("Cannot execute bash commands."), "got: {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // Must fail before ever reaching the runner.
        assert!(runner.last.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn a_cwd_that_is_a_file_not_a_directory_gets_its_own_distinct_message() {
        // Pi-parity fix (task 51): `.is_dir()` correctly rejects a non-directory `cwd` up front (this
        // is actually stricter than pi's own existence-only check), but the error message previously
        // always said "does not exist" even when the path exists and is simply a file. This must be
        // told apart from the genuinely-missing case exercised by
        // `nonexistent_cwd_is_a_clear_invalid_input_error_not_a_raw_spawn_failure` above.
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("not-a-directory.txt");
        std::fs::write(&file_path, "x").unwrap();

        let runner = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        let err = Bash::with_runner(runner.clone())
            .run(json!({ "command": "echo hi", "cwd": file_path.to_str().unwrap() }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("is not a directory"), "got: {msg}");
                assert!(!msg.contains("does not exist"), "got: {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // Must fail before ever reaching the runner.
        assert!(runner.last.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn carriage_returns_in_output_are_preserved_not_stripped() {
        // Deliberate divergence from pi (which strips every `\r`): `is_keepable` explicitly keeps
        // `\r` alongside `\t`/`\n` so progress-bar-style output (`pip`, `npm`, `curl -#`) survives with
        // its real structure intact rather than being silently collapsed. Pinning this so a future
        // change to `is_keepable` can't accidentally start stripping `\r` without a test noticing.
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b[31mred\x1b[0m\r\n".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "red\r\n");
    }

    #[tokio::test]
    async fn a_multibyte_utf8_character_split_across_stream_chunks_decodes_correctly() {
        // pi: tools.test.ts, "should decode UTF-8 characters split across output chunks". `é` is the
        // 2-byte UTF-8 sequence 0xC3 0xA9 — split it across two separate `on_chunk` deliveries and
        // confirm the final text still decodes correctly (raw bytes accumulate before any UTF-8 decode
        // happens, so this should hold by construction; pinning it so a future rewrite can't regress it).
        let runner = Arc::new(ChunkedRunner {
            chunks: vec![b"h\xC3".to_vec(), b"\xA9llo".to_vec()],
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "héllo");
    }

    #[tokio::test]
    async fn streaming_updates_are_throttled_for_chatty_output() {
        // pi: tools.test.ts, "should coalesce streaming updates for chatty output". 5000 tiny chunks
        // arriving back-to-back (no real time passing between them) must not produce 5000 progress
        // events — `UPDATE_THROTTLE` should bound the emitted count to a small number, plus one final
        // flush after the run completes.
        let chunks: Vec<Vec<u8>> = (0..5000).map(|_| b"x".to_vec()).collect();
        let runner = Arc::new(ChunkedRunner { chunks });

        let (tx, mut rx) = futures::channel::mpsc::unbounded();
        let progress = ToolProgress::new(
            tx,
            "id".into(),
            "bash".into(),
            agent_core::CancellationToken::new(),
        );
        Bash::with_runner(runner)
            .run_streaming(json!({ "command": "x" }), &progress)
            .await
            .unwrap();
        drop(progress);

        let mut count = 0usize;
        while rx.try_recv().is_ok() {
            count += 1;
        }
        assert!(
            count < 100,
            "expected a throttled handful of updates for 5000 rapid chunks, got {count}"
        );
    }

    /// A streaming runner that delivers a fixed sequence of chunks with a real delay between each —
    /// unlike `ChunkedRunner` (which fires every chunk back-to-back with no elapsed time, so
    /// `UPDATE_THROTTLE` never trips before the run completes), this lets a test force a live throttled
    /// emit to land at a chosen point mid-stream.
    struct DelayedChunksRunner {
        chunks: Vec<Vec<u8>>,
        delay: Duration,
    }

    #[async_trait]
    impl CommandRunner for DelayedChunksRunner {
        async fn run(
            &self,
            _program: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _timeout: Duration,
        ) -> std::io::Result<ExecResult> {
            unreachable!("test only exercises the streaming path")
        }

        async fn run_streaming(
            &self,
            _program: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _timeout: Duration,
            on_chunk: ChunkSink<'_>,
        ) -> std::io::Result<ExecResult> {
            for (i, chunk) in self.chunks.iter().enumerate() {
                if i > 0 {
                    tokio::time::sleep(self.delay).await;
                }
                on_chunk(chunk);
            }
            Ok(ExecResult {
                code: Some(0),
                ..Default::default()
            })
        }
    }

    #[tokio::test]
    async fn a_live_progress_emit_between_50kb_and_100kb_carries_a_full_output_path() {
        // pi-parity fix (task 58): pi spills to a temp file at 50KB and forces this on *every* live
        // progress emit, not just the final one. This crate's auto-spill during `append` didn't trip
        // until the output outgrew `rolling_cap` (2x the 50KB display budget, i.e. ~100KB), and the
        // live-emit path passed `persist_if_truncated: false` — so a live snapshot taken while the
        // output sat between 50KB and 100KB reported `truncated: true` with no `full_output_path` to
        // point at, even though the final result would have one. Two chunks with a real delay between
        // them (well past `UPDATE_THROTTLE`) force a live emit to land after the first 60KB chunk
        // (over pi's 50KB mark, under the old ~100KB auto-spill mark) and before the run completes.
        let first = vec![b'a'; 60 * 1024];
        let second = b"tail\n".to_vec();
        let runner = Arc::new(DelayedChunksRunner {
            chunks: vec![first, second],
            delay: UPDATE_THROTTLE * 3,
        });

        let (tx, mut rx) = futures::channel::mpsc::unbounded();
        let progress = ToolProgress::new(
            tx,
            "id".into(),
            "bash".into(),
            agent_core::CancellationToken::new(),
        );
        Bash::with_runner(runner)
            .run_streaming(json!({ "command": "x" }), &progress)
            .await
            .unwrap();
        drop(progress);

        // Extract each update's `full_output_path` (or `None` for the initial content-less update),
        // dropping straight to `Option<String>` since `ToolUpdate` itself carries no `Debug` impl to
        // print on assertion failure.
        let mut full_paths = Vec::new();
        while let Ok(update) = rx.try_recv() {
            let agent_core::ToolUpdate::Progress { details, .. } = update else {
                continue;
            };
            let path = details
                .as_ref()
                .and_then(|d| d.get("full_output_path"))
                .and_then(|p| p.as_str())
                .map(str::to_string);
            full_paths.push(path);
        }
        assert!(
            full_paths.len() >= 2,
            "expected at least one live emit plus the final one, got {}",
            full_paths.len()
        );
        // The *last* update is the final snapshot, which was already forced to persist before this
        // fix — the bug is specifically about every update *before* it.
        let (last, live) = full_paths.split_last().expect("checked non-empty above");
        assert!(
            live.iter().any(Option::is_some),
            "a live (pre-final) progress update in the 50-100KB range must already carry a \
             full_output_path, not just the final one: {full_paths:?}"
        );
        assert!(
            last.is_some(),
            "sanity: the final update must still carry a full_output_path too: {full_paths:?}"
        );
    }

    #[tokio::test]
    async fn cancelling_mid_stream_returns_the_output_already_captured_instead_of_discarding_it() {
        // Pi-parity: pi's `bash-executor.ts` returns whatever output already streamed when a command is
        // cancelled (`BashResult { output, cancelled: true }`), not a bare placeholder. Previously this
        // tool's `exec` had no cancellation awareness of its own — a caller could only cancel by
        // dropping the whole future, which discarded everything captured so far, accumulator included.
        let (tx, rx) = futures::channel::mpsc::unbounded();
        let cancel = agent_core::CancellationToken::new();
        let progress = ToolProgress::new(tx, "id".into(), "bash".into(), cancel.clone());

        let bash = Bash::real();
        let run = bash.run_streaming(
            json!({ "command": "printf 'partial-output\\n'; sleep 30" }),
            &progress,
        );
        tokio::pin!(run);
        // Give the `printf` time to actually run and land in the accumulator before cancelling — the
        // command then goes silent for 30s, so nothing further would ever stream via `ToolProgress`
        // (throttled snapshots only fire on new chunk arrival): this is exactly the case an
        // external-drop-only cancellation would have lost entirely.
        tokio::select! {
            _ = &mut run => panic!("must not complete on its own within this window"),
            () = tokio::time::sleep(std::time::Duration::from_millis(300)) => {}
        }
        cancel.cancel();
        let err = tokio::time::timeout(std::time::Duration::from_secs(5), run)
            .await
            .expect("cancellation must resolve promptly, not wait out the sleep")
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("partial-output"),
            "cancellation must preserve output already captured: {msg:?}"
        );
        assert!(
            msg.to_lowercase().contains("cancel"),
            "cancellation result should say so: {msg:?}"
        );
        drop(rx);
    }

    #[tokio::test]
    async fn a_nonexistent_shell_path_produces_a_clear_spawn_error() {
        // pi: tools.test.ts, "should handle process spawn errors". `with_shell_path` pointed at a
        // genuinely nonexistent binary must surface a clear error, not panic or hang.
        let err = Bash::real()
            .with_shell_path("/definitely/not/a/real/shell/binary")
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)), "got: {err:?}");
    }

    #[tokio::test]
    async fn empty_output_timeout_has_no_placeholder_glued_on() {
        // A silent command that times out should read as just the status line, not
        // "(no output)\n\nCommand timed out after Ns" — the "(no output)" placeholder is for the
        // successful/exit-code paths, where an empty result is a genuinely silent command, not an
        // interrupted one.
        let runner = recording(ExecResult {
            timed_out: true,
            stdout: String::new(),
            stderr: String::new(),
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "sleep 10", "timeout_ms": 5000 }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };
        assert_eq!(msg, "Command timed out after 5 seconds");
        assert!(!msg.contains("no output"));
    }
}
