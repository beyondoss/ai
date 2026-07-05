//! Persisted defaults for `run`/`serve` flags (model, gateway URL, session directory) — pi's own
//! `SettingsManager`, scoped down to what this crate's headless CLI surface actually has a use for.
//! Consulted as the last fallback after an explicit `--flag` or its `AI_AGENT_*`/`AI_GATEWAY_URL`
//! environment variable, before this crate's own built-in default — so an operator doesn't have to
//! retype the same flag on every invocation, without needing to export a shell environment variable
//! either. Managed out-of-band via `agent settings` (mirrors `agent trust`/`agent untrust` managing
//! `trust_store.rs`'s allowlist the same way), not through any `run`/`serve` RPC surface — both are
//! one-shot/headless processes with no live session that would need to *change* a stored default mid-run
//! the way pi's long-lived TUI does.
//!
//! Deliberately narrower than pi's own settings file, which also covers TUI-only concerns (theme,
//! terminal rendering, keybindings, external editor integration, telemetry) that don't apply to a
//! headless binary with no interactive UI of its own.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

/// A persisted global default for project trust — pi-parity fix: previously the only durable trust
/// state was the per-path `TrustStore` allowlist (`trust_store.rs`), with no way to set a blanket
/// policy for every project that isn't otherwise explicitly trusted/untrusted. Consulted only when a
/// given invocation passed neither `--trust-project` nor `--force-untrusted` explicitly (see
/// `main.rs`'s trust-resolution call site) — an explicit per-run flag always wins over this global
/// default, the same way an explicit `--model`/`--gateway-url` always wins over `default_model`/
/// `default_gateway_url` above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "snake_case")]
pub enum TrustPolicy {
    /// Treat every project as trusted by default, as if `--trust-project` were always passed.
    Always,
    /// Treat every project as untrusted by default, as if `--force-untrusted` were always passed.
    Never,
    /// No blanket override — fall back to this crate's existing per-path resolution (the persisted
    /// `TrustStore` allowlist, or "no trust-gated resources present at all" — see
    /// `trust_store::has_trust_gated_resources`). This binary is headless, so unlike pi's own
    /// interactive "trust this folder?" prompt, `Ask` doesn't itself ask anything; it just means
    /// "nothing new here, don't override."
    Ask,
}

/// On-disk shape: every field optional and `#[serde(default)]`, so a file missing a field (or not
/// existing at all) degrades to "no stored default" rather than a parse error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// Used when neither `--model` nor `AI_AGENT_MODEL` is given — pi's `defaultModel`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Used when neither `--gateway-url` nor `AI_GATEWAY_URL` is given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_gateway_url: Option<String>,
    /// Used when neither `--session-dir` nor `AI_AGENT_SESSION_DIR` is given — pi's `sessionDir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_session_dir: Option<String>,
    /// Used when neither `--trust-project` nor `--force-untrusted` is given. See [`TrustPolicy`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_project_trust: Option<TrustPolicy>,
    /// A persisted, cross-session override for auto-compaction — Track L26 (pi-parity fix).
    /// `Some(false)` behaves as if `--no-compaction`/`AI_AGENT_NO_COMPACTION` were always passed;
    /// `Some(true)` forces auto-compaction on even if some other layer would otherwise default it off;
    /// `None` (the default) defers entirely to whatever the CLI flag/env var/built-in default already
    /// resolve to. Consulted by `serve`'s `set_auto_compaction` RPC command (see `ServeConfig`'s own doc
    /// comment) so a client's toggle survives a process restart instead of silently reverting to
    /// whatever `--no-compaction` said at the *next* invocation's own startup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_enabled: Option<bool>,
    /// Used when neither `--reasoning-effort`/`--thinking` nor `AI_AGENT_REASONING_EFFORT` is given —
    /// pi-parity fix: `--reasoning-effort` was the only numeric/string CLI tunable with no persisted
    /// stored-default fallback at all (unlike `default_model`/`default_gateway_url`/
    /// `default_session_dir` above). One of `agent_core::ReasoningEffort::as_str`'s wire vocabulary
    /// (minimal/low/medium/high/xhigh) — stored as a plain string (not the enum itself) so this crate's
    /// settings file doesn't take a hard dependency on `agent_core`'s exact type layout, matching how
    /// every other field here is a primitive; validated through the same `parse_reasoning_effort`
    /// `--reasoning-effort` itself uses at both write time (`agent settings --default-reasoning-effort`)
    /// and read time (`main.rs`'s fallback chain), so a corrupt/hand-edited value degrades to "not set"
    /// rather than a panic or a silently-wrong effort reaching the wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_reasoning_effort: Option<String>,
    /// A persisted, cross-session override for whether every image is forced down the same
    /// downgrade-to-text-placeholder path a vision-*incapable* model already gets — Task #26 (pi-parity
    /// feature): Round 1 added `agent_core::Agent::with_block_images`/`Agent.block_images`, an
    /// operator-facing override independent of the active model's real `supports_vision` capability
    /// (bandwidth, compliance, or a proxy that strips/rejects multipart image content, even against a
    /// vision-capable model). `Some(true)` behaves as if `--block-images` were always passed;
    /// `Some(false)`/`None` leave `--block-images`'s own default (images allowed) as the sole source of
    /// truth — mirrors `compaction_enabled`'s identical tri-state convention above. Set via
    /// `agent settings --block-images`/`--clear-block-images`: unlike `compaction_enabled` (mutated via
    /// a live `serve` RPC command), `run` has no long-lived session to toggle it from, so the CLI is the
    /// only mutation surface here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_images: Option<bool>,
    /// Persisted per-reasoning-effort thinking-token-budget overrides — Task #36 (pi-parity feature):
    /// Round 1 added `agent_core::models::budget_for_effort_with_override`, letting a caller tune the
    /// effort-to-token-budget ladder itself rather than only picking an effort level (pi's own model
    /// registry supports the identical per-deployment tuning). Keyed by the same wire vocabulary
    /// `default_reasoning_effort` uses above (minimal/low/medium/high/xhigh); consulted by `main.rs`'s
    /// `run` wherever a turn's thinking budget is derived from `--reasoning-effort`, in place of
    /// `agent_core`'s built-in fixed ladder. Stored as plain strings to u32, not
    /// `agent_core::ReasoningEffort` to u32, matching `default_reasoning_effort`'s own "no hard
    /// dependency on `agent_core`'s exact type layout" convention; an unrecognized key is skipped at
    /// read time rather than failing the whole file (`main.rs::resolve_thinking_budget_overrides`).
    /// `None`/empty are treated the same (no overrides configured at all) — an override table that
    /// loses its last entry is pruned back to `None` rather than left as `Some({})`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget_overrides: Option<std::collections::BTreeMap<String, u32>>,
}

/// A persisted settings file, one per machine (not per-project — matches pi's own global settings tier;
/// this crate has no per-project settings tier to layer on top of it).
pub struct SettingsStore {
    path: PathBuf,
    settings: Settings,
}

impl SettingsStore {
    /// Open the default settings store (`~/.claude/settings.json`). A missing or unparsable file is
    /// treated as all-defaults-unset, never an error.
    pub fn open_default() -> Self {
        Self::open(default_path())
    }

    pub fn open(path: PathBuf) -> Self {
        Self {
            settings: read_store_file(&path),
            path,
        }
    }

    /// The currently stored settings (as of this handle's last read or write — see `mutate_locked` for
    /// why a write always refreshes it against the file's live on-disk state first).
    pub fn get(&self) -> &Settings {
        &self.settings
    }

    /// Set (`Some`) or clear (`None`) the stored default model, persisting atomically.
    pub fn set_default_model(&mut self, model: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_model = model)
    }

    /// Set (`Some`) or clear (`None`) the stored default gateway URL, persisting atomically.
    pub fn set_default_gateway_url(&mut self, url: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_gateway_url = url)
    }

    /// Set (`Some`) or clear (`None`) the stored default session directory, persisting atomically.
    pub fn set_default_session_dir(&mut self, dir: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_session_dir = dir)
    }

    /// Set (`Some`) or clear (`None`) the stored default project-trust policy, persisting atomically.
    pub fn set_default_project_trust(
        &mut self,
        policy: Option<TrustPolicy>,
    ) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_project_trust = policy)
    }

    /// Set (`Some`) or clear (`None`, deferring back to the CLI-flag-computed default) the persisted
    /// auto-compaction override, persisting atomically.
    pub fn set_compaction_enabled(&mut self, enabled: Option<bool>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.compaction_enabled = enabled)
    }

    /// Set (`Some`) or clear (`None`) the stored default reasoning effort, persisting atomically. Takes
    /// the already-validated wire string (`agent_core::ReasoningEffort::as_str`'s vocabulary) — mirrors
    /// `set_default_model`'s identical shape (a plain string, validated by the caller before it ever
    /// reaches here).
    pub fn set_default_reasoning_effort(&mut self, effort: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_reasoning_effort = effort)
    }

    /// Set (`Some`) or clear (`None`) the persisted `--block-images` default, persisting atomically.
    pub fn set_block_images(&mut self, block_images: Option<bool>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.block_images = block_images)
    }

    /// Set (`Some(tokens)`) or clear (`None`) the persisted thinking-token-budget override for one
    /// reasoning-effort level (`effort`: a wire string — validated by the caller, matching
    /// `default_reasoning_effort`'s own plain-string convention so this store doesn't need a hard
    /// dependency on `agent_core`'s enum layout), persisting atomically. An override table that becomes
    /// empty after clearing its last entry is pruned back to `None` rather than left as `Some({})`,
    /// mirroring every other optional field's "unset looks unset" convention.
    pub fn set_thinking_budget_override(
        &mut self,
        effort: String,
        tokens: Option<u32>,
    ) -> std::io::Result<()> {
        self.mutate_locked(move |s| {
            let table = s.thinking_budget_overrides.get_or_insert_with(Default::default);
            match tokens {
                Some(t) => {
                    table.insert(effort, t);
                }
                None => {
                    table.remove(&effort);
                }
            }
            if table.is_empty() {
                s.thinking_budget_overrides = None;
            }
        })
    }

    /// Acquire the cross-process lock (see [`FileLock`]), re-read the store's *current* on-disk state —
    /// not `self`'s possibly-stale in-memory copy, which another process may have moved past since this
    /// handle's `open()` or its own last mutation — apply `mutate` to that fresh state, and persist it.
    /// `self` is updated to match afterward, so this handle's own `get()` stays consistent with what's
    /// now on disk. Mirrors `trust_store.rs::TrustStore::mutate_locked`'s identical reasoning: two
    /// `agent settings` invocations changing different fields concurrently must not let whichever writes
    /// last silently clobber the other's change.
    fn mutate_locked(&mut self, mutate: impl FnOnce(&mut Settings)) -> std::io::Result<()> {
        let _lock = FileLock::acquire(&self.path)?;
        let mut settings = read_store_file(&self.path);
        mutate(&mut settings);
        write_store_file(&self.path, &settings)?;
        self.settings = settings;
        Ok(())
    }
}

/// Read and parse the store file at `path`, tolerating a missing or unparsable file (all-unset) — shared
/// by [`SettingsStore::open`] and [`SettingsStore::mutate_locked`], which both need "current on-disk
/// state, gracefully defaulted."
fn read_store_file(path: &Path) -> Settings {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_store_file(path: &Path, settings: &Settings) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(settings).map_err(std::io::Error::other)?;
    let path_str = path
        .to_str()
        .ok_or_else(|| std::io::Error::other(format!("non-UTF-8 path: {}", path.display())))?;
    crate::tools::write_atomic(path_str, body.as_bytes())
}

/// How long [`FileLock::acquire`] retries before giving up.
const LOCK_TIMEOUT: Duration = Duration::from_secs(5);
/// How long between retries while waiting for another process's lock.
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(20);
/// A lock file older than this is assumed abandoned (its owner crashed without cleaning up, rather than
/// genuinely being mid-operation for this long — a settings read/parse/write is a handful of
/// milliseconds) and is forcibly reclaimed instead of blocking every future `agent settings` indefinitely.
const STALE_LOCK_AGE: Duration = Duration::from_secs(10);

/// A cross-process advisory lock via atomic lockfile creation — the same pattern
/// `trust_store.rs::FileLock` already uses for its own on-disk store, duplicated here rather than shared:
/// each store is small and self-contained, and the two have no other reason to depend on each other.
/// Released by deleting the lock file on `Drop`, so a panicked or early-returning holder still releases
/// it.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(store_path: &Path) -> std::io::Result<Self> {
        let lock_path = lock_path_for(store_path);
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let deadline = Instant::now() + LOCK_TIMEOUT;
        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_) => return Ok(Self { path: lock_path }),
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    if is_stale(&lock_path) {
                        // Best-effort: another process reclaiming it at the same instant just means our
                        // own retry loop's next `create_new` fails normally and we wait it out.
                        let _ = fs::remove_file(&lock_path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(std::io::Error::new(
                            ErrorKind::TimedOut,
                            format!(
                                "timed out waiting for settings store lock at {}",
                                lock_path.display()
                            ),
                        ));
                    }
                    std::thread::sleep(LOCK_RETRY_INTERVAL);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn is_stale(lock_path: &Path) -> bool {
    fs::metadata(lock_path)
        .and_then(|m| m.modified())
        .and_then(|modified| {
            SystemTime::now()
                .duration_since(modified)
                .map_err(std::io::Error::other)
        })
        .is_ok_and(|age| age > STALE_LOCK_AGE)
}

fn lock_path_for(store_path: &Path) -> PathBuf {
    let mut os = store_path.as_os_str().to_owned();
    os.push(".lock");
    PathBuf::from(os)
}

/// The root config directory both this store and `trust_store.rs`'s persist under — `~/.claude` unless
/// overridden. pi-parity fix: previously hardcoded at each call site (here and in `trust_store.rs`),
/// with no way to redirect it short of overriding the whole `$HOME` (which also moves every *other*
/// HOME-relative thing — shells, other tools' own configs). `AI_AGENT_CONFIG_DIR`, when set, names the
/// config directory itself directly (not a parent to append `.claude` under), so a caller who wants
/// a wholly separate profile/sandbox doesn't have to also fake up a `.claude` subdirectory inside it.
/// `session_store.rs` hardcodes the same `~/.claude/sessions` root independently (a separate agent's own
/// file to update) — this helper is `pub(crate)` specifically so that call site can adopt it too, later,
/// without needing to duplicate the env-var-vs-`HOME` precedence logic a second time.
pub(crate) fn config_dir_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("AI_AGENT_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".claude")
}

fn default_path() -> PathBuf {
    config_dir_root().join("settings.json")
}

/// A single model's on-disk connection override — pi-parity feature (Fix 9): pi's own
/// `model-registry.ts` lets an operator override a model's base URL/auth/routing preferences via a
/// `models.json`, merged over its built-in provider catalog. This crate holds no catalog of its own to
/// merge over (a model id is forwarded verbatim through the gateway — see `agent_core::models`'s own
/// module doc comment), so there's nothing to "merge": an override here just redirects where a matching
/// model id's request actually goes, consulted by `main.rs::resolve_gateway_credential`. Deliberately
/// narrower than pi's own schema — no OpenRouter-specific routing-preference surface
/// (order/only/ignore/zdr/sort/max_price/...): a plain base-url/auth override per model id is the
/// valuable core capability, and this crate has no OpenRouter-specific request shaping to hang routing
/// preferences off of in the first place.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelOverride {
    /// Send this model's requests directly here instead of the gateway's own default per-dialect
    /// routing — bypasses the gateway entirely, using the model's usual wire path
    /// (`agent_core::dialect::Dialect::endpoint_path`) appended to this base URL. The same
    /// `agent_core::client::RouteOverride::Direct` mechanism already used for a GitHub-Copilot-routed
    /// OAuth login (see `main.rs::resolve_gateway_credential`), reused here rather than duplicated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// A bearer token to send instead of whatever `--key`/`AI_AGENT_KEY` resolved to — e.g. a distinct
    /// key for a self-hosted or alternate-provider endpoint. `None` falls back to `--key`/
    /// `AI_AGENT_KEY` (empty string if neither is set — many self-hosted OpenAI-compatible servers, like
    /// Ollama/LM Studio, ignore the `Authorization` header entirely, so this is a usable default rather
    /// than a hard error). May be a literal, a `!command` (the rest of the string run as a shell
    /// command, trimmed stdout used), or a `$VAR`/`${VAR}` environment-variable reference — see
    /// [`resolve_config_value`] — so an operator doesn't have to store a plaintext secret in
    /// `models.json`. Resolved via [`Self::resolved_api_key`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Extra headers to send on every request to this model, merged in via
    /// `agent_core::client::GatewayClient::with_extra_headers` (Task #11 pi-parity feature) — a custom
    /// proxy auth header, a tracing header, anything unrelated to the primary credential — matching pi's
    /// own `model-registry.ts` `headers` config. Independent of `base_url`: a header-only override (no
    /// `base_url` at all) still routes through the gateway as usual, with these headers merged on top.
    /// Each value resolves through the same [`resolve_config_value`] syntax `api_key` does — a literal,
    /// a `!command`, or a `$VAR`/`${VAR}` reference. Resolved via [`Self::resolved_headers`].
    ///
    /// For a *direct* (bypassing-the-gateway, `base_url`-set) override whose endpoint needs a non-Bearer
    /// auth scheme — Azure OpenAI's `api-key: <key>` instead of `Authorization: Bearer <key>` being the
    /// motivating case — use [`Self::auth_header`] instead of stuffing the credential in here: `headers`
    /// is merged *alongside* whatever `Authorization` header `base_url`/`api_key` produces, so putting a
    /// duplicate credential here doesn't stop that `Authorization: Bearer` header (carrying either
    /// `api_key` or, if unset, a silent fallback to `--key`/`AI_AGENT_KEY` — a real risk of leaking the
    /// gateway's own virtual key to an unrelated third-party endpoint) from also being sent.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub headers: std::collections::BTreeMap<String, String>,
    /// For a *direct* (`base_url`-set) override: send `api_key`'s resolved value through this header
    /// name instead of `Authorization: Bearer <key>` — e.g. `"api-key"` for Azure OpenAI, which never
    /// accepts a Bearer token and whose SDK omits `Authorization` entirely. `None` (the default) keeps
    /// the existing `Authorization: Bearer` behavior. Has no effect without `base_url` set — a
    /// gateway-routed override's auth scheme is the gateway's concern, not the client's. Threaded to
    /// `agent_core::client::DirectRouting::auth_header` in `main.rs::resolve_gateway_credential`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<String>,
}

impl ModelOverride {
    /// Resolve this override's `api_key` (if any) through [`resolve_config_value`] — falling back to
    /// `fallback` (whatever `--key`/`AI_AGENT_KEY` resolved to) when `api_key` is unset, unresolvable
    /// (an unset env var, a failing `!command`), or resolves to an empty string, then finally to `""`
    /// when neither is usable — many self-hosted OpenAI-compatible servers ignore the `Authorization`
    /// header entirely, so an empty bearer is a usable default rather than a hard error (matches this
    /// struct's own prior literal-`api_key` convention).
    pub fn resolved_api_key(&self, fallback: Option<&str>) -> String {
        self.resolved_api_key_with_env(fallback, None)
    }

    /// [`Self::resolved_api_key`], but with an explicit environment-variable override map consulted
    /// before the real process environment — production callers go through the public method (`None`
    /// here, real environment only); tests inject a fixed map instead of mutating the real,
    /// process-wide (and test-parallelism-unsafe — see `resources.rs`'s identical `$HOME` concern)
    /// environment.
    fn resolved_api_key_with_env(
        &self,
        fallback: Option<&str>,
        env_overrides: Option<&std::collections::HashMap<String, String>>,
    ) -> String {
        self.api_key
            .as_deref()
            .and_then(|v| resolve_config_value(v, env_overrides))
            .filter(|v| !v.is_empty())
            .or_else(|| fallback.map(str::to_string))
            .unwrap_or_default()
    }

    /// Resolve every configured header value through [`resolve_config_value`], dropping any entry that
    /// fails to resolve (an unset env var, a failing `!command`) rather than sending a literal `$VAR`/
    /// `!command` string as a header value.
    pub fn resolved_headers(&self) -> std::collections::HashMap<String, String> {
        self.resolved_headers_with_env(None)
    }

    /// [`Self::resolved_headers`], but with an explicit environment-variable override map — see
    /// [`Self::resolved_api_key_with_env`]'s doc comment for why tests use this instead of the public
    /// method.
    fn resolved_headers_with_env(
        &self,
        env_overrides: Option<&std::collections::HashMap<String, String>>,
    ) -> std::collections::HashMap<String, String> {
        self.headers
            .iter()
            .filter_map(|(k, v)| resolve_config_value(v, env_overrides).map(|v| (k.clone(), v)))
            .collect()
    }
}

/// Resolve a `models.json` `api_key`/header config value that may be a literal, an environment-variable
/// reference (`$VAR`/`${VAR}`, with `$$`/`$!` escaping a literal `$`/`!`), or a shell command
/// (`!command`) — pi's own `resolveConfigValue`
/// (`packages/coding-agent/src/core/resolve-config-value.ts`), the exact mechanism its
/// `model-registry.ts` already uses for this (a self-hosted/corporate-proxied endpoint's non-Bearer
/// auth, or keeping a plaintext secret out of `models.json`). Ported down to what this crate's
/// `ModelOverride` needs: no process-lifetime command-result cache (each override is resolved once, at
/// gateway-client-construction time, not per-request, so there's nothing to amortize), and no
/// configured-shell/Windows fallback (this binary only ships for Unix — see `tools::bash`'s own
/// shell-resolution precedent).
///
/// `env_overrides`, when given, is consulted before the real process environment for every `$VAR`/
/// `${VAR}` reference — production callers pass `None` (real environment only); tests inject a fixed
/// map instead of mutating the real, process-wide environment (see `ModelOverride::
/// resolved_api_key_with_env`'s doc comment).
///
/// - A value starting with `!` runs the rest as a shell command (`sh -c`) and uses its trimmed stdout; a
///   nonzero exit or empty stdout resolves to `None`.
/// - Otherwise, `$VAR`/`${VAR}` references are substituted with that environment variable's value; `$$`
///   escapes a literal `$`, `$!` escapes a literal `!`. Any referenced variable that's unset resolves
///   the *whole* value to `None` — never a partial/garbled substitution.
/// - A value with no `$`/`!` markup at all resolves to itself, unchanged.
fn resolve_config_value(
    config: &str,
    env_overrides: Option<&std::collections::HashMap<String, String>>,
) -> Option<String> {
    if let Some(command) = config.strip_prefix('!') {
        return run_config_value_command(command);
    }
    let lookup = |name: &str| -> Option<String> {
        env_overrides
            .and_then(|m| m.get(name).cloned())
            .or_else(|| std::env::var(name).ok())
    };
    resolve_config_value_template(config, &lookup)
}

/// The `$VAR`/`${VAR}`/`$$`/`$!` template half of [`resolve_config_value`] — split out so its
/// character-by-character scan is unit-testable independent of the `!command`/env-lookup plumbing
/// around it.
fn resolve_config_value_template(config: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    let chars: Vec<char> = config.chars().collect();
    let mut out = String::with_capacity(config.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '$' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        match chars.get(i + 1).copied() {
            Some('$') => {
                out.push('$');
                i += 2;
            }
            Some('!') => {
                out.push('!');
                i += 2;
            }
            Some('{') => match chars[i + 2..].iter().position(|&c| c == '}') {
                Some(rel) => {
                    let end = i + 2 + rel;
                    let name: String = chars[i + 2..end].iter().collect();
                    if is_env_var_name(&name) {
                        out.push_str(&lookup(&name)?);
                    } else {
                        out.extend(chars[i..=end].iter());
                    }
                    i = end + 1;
                }
                None => {
                    out.push('$');
                    i += 1;
                }
            },
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let start = i + 1;
                let mut end = start;
                while end < chars.len() && (chars[end].is_ascii_alphanumeric() || chars[end] == '_') {
                    end += 1;
                }
                let name: String = chars[start..end].iter().collect();
                out.push_str(&lookup(&name)?);
                i = end;
            }
            _ => {
                out.push('$');
                i += 1;
            }
        }
    }
    Some(out)
}

/// Whether `s` is a valid shell-style environment-variable name (`[A-Za-z_][A-Za-z0-9_]*`) — gates
/// `${...}` template substitution so e.g. `${not a var}` is left as a literal instead of silently
/// resolving to `None` (matches pi's own `ENV_VAR_NAME_RE`).
fn is_env_var_name(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Run `command` via `sh -c` and return its trimmed stdout, or `None` on a nonzero exit, empty stdout,
/// or a failure to even spawn the shell (matches pi's own `executeCommand`'s "unresolvable" semantics).
fn run_config_value_command(command: &str) -> Option<String> {
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// On-disk shape: a flat JSON object of model id → [`ModelOverride`] (`{"my-model": {"base_url":
/// "http://127.0.0.1:11434"}}`) — the minimal "array/map of model-id → overrides" schema this feature
/// calls for, not pi's nested `{"providers": {...}}` structure (this crate has no provider concept of
/// its own to nest under; see [`ModelOverride`]'s own doc comment).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelOverrides(std::collections::BTreeMap<String, ModelOverride>);

impl ModelOverrides {
    /// Load the default model-override file (`~/.claude/models.json`, alongside `settings.json`/
    /// `trusted-projects.json` — see [`config_dir_root`]). A missing file is the ordinary case (no
    /// overrides configured at all) — silent, matching [`SettingsStore::open`]'s own convention. An
    /// existing-but-unparsable file `warn!`s and is treated as empty rather than failing the whole
    /// `run`/`serve` invocation over a hand-edited typo — mirrors `trust_store.rs`'s identical "skip and
    /// warn, don't silently lose data, but don't hard-fail a normal invocation either" precedent for its
    /// own corrupt-store case.
    pub fn open_default() -> Self {
        Self::open(model_overrides_path())
    }

    /// Same as [`Self::open_default`], but against an arbitrary path — for tests, and for any future
    /// caller that wants to point at a non-default location.
    pub fn open(path: PathBuf) -> Self {
        let contents = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == ErrorKind::NotFound => return Self::default(),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "could not read model overrides file, treating it as empty"
                );
                return Self::default();
            }
        };
        match serde_json::from_str(&contents) {
            Ok(overrides) => overrides,
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "model overrides file is not valid JSON, treating it as empty"
                );
                Self::default()
            }
        }
    }

    /// The override recorded for `model_id`, if any.
    pub fn get(&self, model_id: &str) -> Option<&ModelOverride> {
        self.0.get(model_id)
    }

    /// How many model ids have an override configured — surfaced by `agent settings` so an operator can
    /// tell this file is actually being read (this feature's "CLI-visible" requirement) without a
    /// dedicated dump-the-whole-file command.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether no model ids have an override configured (including "the file doesn't exist at all").
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Where [`ModelOverrides::open_default`] reads from — a sibling of [`default_path`]'s
/// `settings.json`/`trust_store.rs`'s `trusted-projects.json`, under the same [`config_dir_root`].
pub fn model_overrides_path() -> PathBuf {
    config_dir_root().join("models.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_store_file_is_all_defaults_unset() {
        let dir = tempfile::tempdir().unwrap();
        let store = SettingsStore::open(dir.path().join("does-not-exist.json"));
        assert_eq!(store.get(), &Settings::default());
    }

    #[test]
    fn set_default_model_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        store
            .set_default_model(Some("claude-opus-4-8".to_string()))
            .unwrap();
        assert_eq!(
            store.get().default_model.as_deref(),
            Some("claude-opus-4-8")
        );

        let reopened = SettingsStore::open(path);
        assert_eq!(
            reopened.get().default_model.as_deref(),
            Some("claude-opus-4-8")
        );
    }

    #[test]
    fn set_compaction_enabled_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(
            store.get().compaction_enabled,
            None,
            "no stored override by default"
        );

        store.set_compaction_enabled(Some(false)).unwrap();
        assert_eq!(store.get().compaction_enabled, Some(false));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().compaction_enabled, Some(false));

        // Clearing it back to `None` (not just flipping to `true`) must also round-trip.
        let mut store = SettingsStore::open(path.clone());
        store.set_compaction_enabled(None).unwrap();
        assert_eq!(store.get().compaction_enabled, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().compaction_enabled, None);
    }

    #[test]
    fn clearing_a_field_removes_it_without_touching_others() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        store
            .set_default_model(Some("claude-opus-4-8".to_string()))
            .unwrap();
        store
            .set_default_gateway_url(Some("http://gw.internal".to_string()))
            .unwrap();

        store.set_default_model(None).unwrap();
        assert_eq!(store.get().default_model, None);
        assert_eq!(
            store.get().default_gateway_url.as_deref(),
            Some("http://gw.internal"),
            "clearing one field must not clobber another already-set field"
        );

        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_model, None);
        assert_eq!(
            reopened.get().default_gateway_url.as_deref(),
            Some("http://gw.internal")
        );
    }

    #[test]
    fn a_stale_in_memory_snapshot_is_refreshed_before_mutating_not_blindly_overwritten() {
        // Two handles on the same file (matching what actually happens across two separate `agent
        // settings` invocations) each setting a *different* field must not let one silently lose the
        // other's write — `handle_b` is opened before `handle_a`'s write, so its in-memory state is
        // genuinely stale by the time it mutates.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");

        let mut handle_a = SettingsStore::open(path.clone());
        let mut handle_b = SettingsStore::open(path.clone());

        handle_a
            .set_default_model(Some("model-a".to_string()))
            .unwrap();
        handle_b
            .set_default_session_dir(Some("/dir-b".to_string()))
            .unwrap();

        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_model.as_deref(), Some("model-a"));
        assert_eq!(
            reopened.get().default_session_dir.as_deref(),
            Some("/dir-b")
        );
    }

    #[test]
    fn store_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        // A nested store path whose parent directory doesn't exist yet — matches the real
        // `~/.claude/settings.json` layout on a fresh `~/.claude`.
        let path = dir.path().join("nested/.claude/settings.json");
        let mut store = SettingsStore::open(path.clone());
        store.set_default_model(Some("m".to_string())).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn a_stale_lock_file_is_reclaimed_rather_than_blocking_forever() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let lock_path = lock_path_for(&path);
        fs::write(&lock_path, b"").unwrap();
        let stale = SystemTime::now() - (STALE_LOCK_AGE + Duration::from_secs(1));
        fs::File::open(&lock_path)
            .unwrap()
            .set_modified(stale)
            .unwrap();

        let mut store = SettingsStore::open(path);
        store.set_default_model(Some("m".to_string())).unwrap();
        assert_eq!(store.get().default_model.as_deref(), Some("m"));
    }

    #[test]
    fn unrecognized_fields_in_the_file_are_ignored_not_a_parse_error() {
        // Forward-compatibility: a settings file written by a future version with extra fields must
        // still open cleanly here, not fail outright.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(
            &path,
            r#"{"default_model":"m","theme":"dark","enableAnalytics":true}"#,
        )
        .unwrap();
        let store = SettingsStore::open(path);
        assert_eq!(store.get().default_model.as_deref(), Some("m"));
    }

    #[test]
    fn set_default_reasoning_effort_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_reasoning_effort, None);

        store
            .set_default_reasoning_effort(Some("high".to_string()))
            .unwrap();
        assert_eq!(
            store.get().default_reasoning_effort.as_deref(),
            Some("high")
        );

        let reopened = SettingsStore::open(path.clone());
        assert_eq!(
            reopened.get().default_reasoning_effort.as_deref(),
            Some("high")
        );

        // Clearing it back to `None` must also round-trip, mirroring every other field's convention.
        let mut store = SettingsStore::open(path.clone());
        store.set_default_reasoning_effort(None).unwrap();
        assert_eq!(store.get().default_reasoning_effort, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_reasoning_effort, None);
    }

    #[test]
    fn missing_model_overrides_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let overrides = ModelOverrides::open(dir.path().join("does-not-exist.json"));
        assert!(overrides.is_empty());
        assert_eq!(overrides.get("anything"), None);
    }

    #[test]
    fn model_overrides_file_round_trips_a_base_url_and_api_key_override() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        fs::write(
            &path,
            r#"{"my-local-model": {"base_url": "http://127.0.0.1:11434", "api_key": "local-key"}}"#,
        )
        .unwrap();

        let overrides = ModelOverrides::open(path);
        assert_eq!(overrides.len(), 1);
        let over = overrides.get("my-local-model").unwrap();
        assert_eq!(over.base_url.as_deref(), Some("http://127.0.0.1:11434"));
        assert_eq!(over.api_key.as_deref(), Some("local-key"));
        assert_eq!(overrides.get("some-other-model"), None);
    }

    #[test]
    fn a_corrupt_model_overrides_file_warns_and_is_treated_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        fs::write(&path, "not valid json at all { [ }").unwrap();

        let capture = crate::tracing_test::capture(|| {
            let overrides = ModelOverrides::open(path);
            assert!(overrides.is_empty());
        });
        assert!(
            capture
                .messages()
                .iter()
                .any(|m| m.contains("model overrides")),
            "a corrupt model overrides file must be logged, not silently swallowed: {:?}",
            capture.messages()
        );
    }

    #[test]
    fn model_overrides_default_path_is_a_sibling_of_settings_json() {
        assert_eq!(
            model_overrides_path().file_name().unwrap(),
            "models.json"
        );
        assert_eq!(
            model_overrides_path().parent(),
            default_path().parent()
        );
    }

    #[test]
    fn set_block_images_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().block_images, None);

        store.set_block_images(Some(true)).unwrap();
        assert_eq!(store.get().block_images, Some(true));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().block_images, Some(true));

        let mut store = SettingsStore::open(path.clone());
        store.set_block_images(None).unwrap();
        assert_eq!(store.get().block_images, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().block_images, None);
    }

    #[test]
    fn set_thinking_budget_override_persists_a_single_entry_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().thinking_budget_overrides, None);

        store
            .set_thinking_budget_override("high".to_string(), Some(40_000))
            .unwrap();
        assert_eq!(
            store.get().thinking_budget_overrides.as_ref().unwrap().get("high"),
            Some(&40_000)
        );
        let reopened = SettingsStore::open(path);
        assert_eq!(
            reopened
                .get()
                .thinking_budget_overrides
                .as_ref()
                .unwrap()
                .get("high"),
            Some(&40_000)
        );
    }

    #[test]
    fn set_thinking_budget_override_accumulates_multiple_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        store
            .set_thinking_budget_override("high".to_string(), Some(40_000))
            .unwrap();
        store
            .set_thinking_budget_override("low".to_string(), Some(1_000))
            .unwrap();

        let table = store.get().thinking_budget_overrides.as_ref().unwrap();
        assert_eq!(table.get("high"), Some(&40_000));
        assert_eq!(table.get("low"), Some(&1_000));
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn set_thinking_budget_override_clearing_the_last_entry_prunes_the_table_back_to_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        store
            .set_thinking_budget_override("high".to_string(), Some(40_000))
            .unwrap();
        store
            .set_thinking_budget_override("high".to_string(), None)
            .unwrap();
        assert_eq!(
            store.get().thinking_budget_overrides,
            None,
            "clearing the only entry must prune the table back to None, not leave Some({{}})"
        );
    }

    #[test]
    fn model_overrides_file_round_trips_a_header_only_override_with_no_base_url_or_api_key() {
        // Task #11: a header-only override (e.g. Azure's `api-key:` header) must still route through
        // the gateway as usual — no `base_url`/`api_key` needed at all for `headers` to be useful.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        fs::write(
            &path,
            r#"{"azure-gpt": {"headers": {"api-key": "my-azure-key"}}}"#,
        )
        .unwrap();

        let overrides = ModelOverrides::open(path);
        let over = overrides.get("azure-gpt").unwrap();
        assert_eq!(over.base_url, None);
        assert_eq!(over.api_key, None);
        assert_eq!(
            over.resolved_headers().get("api-key").map(String::as_str),
            Some("my-azure-key")
        );
    }

    #[test]
    fn resolved_api_key_runs_a_bang_prefixed_command_and_uses_its_trimmed_stdout() {
        let over = ModelOverride {
            api_key: Some("!echo -n my-secret-123".to_string()),
            ..Default::default()
        };
        assert_eq!(over.resolved_api_key(None), "my-secret-123");
    }

    #[test]
    fn resolved_api_key_falls_back_when_the_command_fails() {
        let over = ModelOverride {
            api_key: Some("!exit 1".to_string()),
            ..Default::default()
        };
        assert_eq!(over.resolved_api_key(Some("fallback-key")), "fallback-key");
        assert_eq!(over.resolved_api_key(None), "");
    }

    #[test]
    fn resolved_api_key_falls_back_to_the_operator_key_when_unset() {
        let over = ModelOverride::default();
        assert_eq!(over.resolved_api_key(Some("operator-key")), "operator-key");
        assert_eq!(over.resolved_api_key(None), "");
    }

    #[test]
    fn resolved_api_key_uses_the_literal_value_unchanged_when_it_has_no_markup() {
        let over = ModelOverride {
            api_key: Some("plain-literal-key".to_string()),
            ..Default::default()
        };
        assert_eq!(over.resolved_api_key(None), "plain-literal-key");
    }

    #[test]
    fn resolved_headers_substitutes_a_dollar_var_reference_from_the_injected_env_map() {
        let mut over = ModelOverride::default();
        over.headers.insert(
            "Authorization".to_string(),
            "Bearer $MY_PROXY_TOKEN".to_string(),
        );
        let mut env = std::collections::HashMap::new();
        env.insert("MY_PROXY_TOKEN".to_string(), "tok-abc-123".to_string());

        let resolved = over.resolved_headers_with_env(Some(&env));
        assert_eq!(
            resolved.get("Authorization").map(String::as_str),
            Some("Bearer tok-abc-123")
        );
    }

    #[test]
    fn resolved_headers_substitutes_a_braced_dollar_var_reference() {
        let mut over = ModelOverride::default();
        over.headers
            .insert("X-Proxy-Key".to_string(), "${PROXY_KEY}-suffix".to_string());
        let mut env = std::collections::HashMap::new();
        env.insert("PROXY_KEY".to_string(), "abc".to_string());

        let resolved = over.resolved_headers_with_env(Some(&env));
        assert_eq!(
            resolved.get("X-Proxy-Key").map(String::as_str),
            Some("abc-suffix")
        );
    }

    #[test]
    fn resolved_headers_drops_an_entry_whose_referenced_env_var_is_unset() {
        let mut over = ModelOverride::default();
        over.headers
            .insert("X-Missing".to_string(), "$TOTALLY_UNSET_VAR_XYZ".to_string());
        over.headers
            .insert("X-Present".to_string(), "literal".to_string());

        let resolved = over.resolved_headers_with_env(Some(&std::collections::HashMap::new()));
        assert_eq!(resolved.get("X-Missing"), None);
        assert_eq!(resolved.get("X-Present").map(String::as_str), Some("literal"));
    }

    #[test]
    fn auth_header_defaults_to_none_and_round_trips_through_json() {
        let over = ModelOverride::default();
        assert_eq!(over.auth_header, None);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        std::fs::write(
            &path,
            r#"{"azure-gpt": {"base_url": "https://foo.openai.azure.com/openai/deployments/x", "auth_header": "api-key", "api_key": "my-azure-key"}}"#,
        )
        .unwrap();

        let overrides = ModelOverrides::open(path);
        let over = overrides.get("azure-gpt").unwrap();
        assert_eq!(over.auth_header.as_deref(), Some("api-key"));
        assert_eq!(over.resolved_api_key(None), "my-azure-key");
    }

    #[test]
    fn auth_header_is_omitted_from_serialized_json_when_unset() {
        let over = ModelOverride {
            base_url: Some("https://example.com".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&over).unwrap();
        assert!(
            !json.contains("auth_header"),
            "unset auth_header must not appear in the serialized form, got: {json}"
        );
    }

    #[test]
    fn resolve_config_value_template_escapes_dollar_and_bang() {
        let lookup = |_: &str| -> Option<String> { None };
        assert_eq!(
            resolve_config_value_template("price: $$5 $!important", &lookup),
            Some("price: $5 !important".to_string())
        );
    }

    #[test]
    fn resolve_config_value_command_form_takes_precedence_over_template_parsing() {
        // A `!`-prefixed value is a command outright, even though `!` alone (unescaped, not at the
        // start) is otherwise just a literal character in template parsing.
        assert_eq!(
            resolve_config_value("!echo -n hello", None),
            Some("hello".to_string())
        );
    }

    #[test]
    fn is_env_var_name_rejects_a_name_starting_with_a_digit_or_containing_a_space() {
        assert!(!is_env_var_name("1BAD"));
        assert!(!is_env_var_name("not a var"));
        assert!(is_env_var_name("_valid_Name9"));
    }
}
