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
    /// `default_session_dir` above). One of `agent_core::ThinkingLevel::as_str`'s wire vocabulary
    /// (off/minimal/low/medium/high/xhigh — Task 2, pi-parity fix pass 19: `"off"` is now a legal value
    /// here too, matching pi's own `--thinking off` and this crate's RPC-facing `set_reasoning_effort`,
    /// both of which already treated "explicitly disable reasoning" as a first-class request rather than
    /// a parse error) — stored as a plain string (not the enum itself) so this crate's settings file
    /// doesn't take a hard dependency on `agent_core`'s exact type layout, matching how every other field
    /// here is a primitive; validated through the same `parse_reasoning_effort` `--reasoning-effort`
    /// itself uses at both write time (`agent settings --default-reasoning-effort`) and read time
    /// (`main.rs`'s fallback chain), so a corrupt/hand-edited value degrades to "not set" rather than a
    /// panic or a silently-wrong effort reaching the wire.
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
    /// A persisted, cross-session override for whether `read` downscales/re-encodes an oversized image
    /// to fit the inline size budget — Task #4 (pi-parity feature): pi's own `ImageSettings.autoResize`
    /// (default `true`). `Some(false)` behaves as if `--no-image-auto-resize` were always passed;
    /// `Some(true)`/`None` leave the built-in default (resize enabled) in effect — mirrors
    /// `block_images`'s identical tri-state convention just above, just inverted (this one defaults to
    /// *on*, `block_images` defaults to *off*). Set via `agent settings --image-auto-resize`/
    /// `--clear-image-auto-resize`: like `block_images`, `run` has no long-lived session to toggle this
    /// from, so the CLI is the only mutation surface here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_auto_resize: Option<bool>,
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
    /// Used when neither `--bash-shell-path` nor `AI_AGENT_BASH_SHELL_PATH` is given, for both `run` and
    /// `serve` — pi-parity fix (Round 3): previously flag/env-only, with no persisted stored-default
    /// fallback at all, unlike `default_model`/`default_gateway_url`/`default_session_dir` above.
    /// Consulted at the same fallback-chain position those fields already occupy (explicit flag/env,
    /// then this stored setting, then the auto-resolved shell — `/bin/bash`, else `bash` on `$PATH`,
    /// else `sh`). Not itself re-validated to exist by this store — `serve`'s existing
    /// `--bash-shell-path`-not-found startup check runs against whichever value wins the whole chain,
    /// exactly as it would for an explicit flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_bash_shell_path: Option<String>,
    /// Used when neither `--bash-command-prefix` nor `AI_AGENT_BASH_COMMAND_PREFIX` is given —
    /// pi-parity fix (Round 3), [`Self::default_bash_shell_path`]'s identical fallback rationale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_bash_command_prefix: Option<String>,
    /// Used when neither `--compaction-reserve-tokens` nor `AI_AGENT_COMPACTION_RESERVE_TOKENS` is
    /// given — pi-parity fix (Round 3): an operator-persisted override of one layer *below* this,
    /// `agent_core::CompactionConfig::default()`'s own built-in 16,384-token reserve — the same
    /// "override the built-in default, don't replace it outright" relationship
    /// [`Self::default_reasoning_effort`] has to `agent_core`'s own reasoning default. `None` leaves
    /// `CompactionConfig::default()`'s value in effect, unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_compaction_reserve_tokens: Option<u32>,
    /// Used when neither `--compaction-keep-recent-tokens` nor
    /// `AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS` is given — pi-parity fix (Round 3),
    /// [`Self::default_compaction_reserve_tokens`]'s identical relationship to
    /// `CompactionConfig::default()`'s own built-in 20,000-token `keep_recent_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_compaction_keep_recent_tokens: Option<u32>,
    /// Used when neither `--retry-max-retries` nor `AI_AGENT_RETRY_MAX_RETRIES` is given — pi-parity fix
    /// (Round 3): previously flag/env-only, unlike `default_model` et al. above. `None` leaves both
    /// layers this flag already drives (`agent_core::client::MAX_RETRIES`'s pre-first-byte/mid-stream
    /// retry, and `retry::MAX_RUN_RETRIES`'s whole-run retry — see `run_task`'s own `with_retry`/
    /// `RunRetryPolicy::from_overrides` call sites) at their built-in defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_retry_max_retries: Option<u32>,
    /// Used when neither `--retry-base-delay-ms` nor `AI_AGENT_RETRY_BASE_DELAY_MS` is given —
    /// pi-parity fix (Round 3), [`Self::default_retry_max_retries`]'s identical relationship to
    /// `agent_core::client::BASE_BACKOFF`/`retry::RUN_RETRY_BASE_BACKOFF`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_retry_base_delay_ms: Option<u64>,
    /// Used when neither `--idle-timeout-ms` nor `AI_AGENT_IDLE_TIMEOUT_MS` is given — this crate's
    /// "provider timeout": the idle-read timeout between response chunks on the gateway HTTP client (see
    /// `run`'s `--idle-timeout-ms` doc comment, Task #19) — pi-parity fix (Round 3): the last of the
    /// retry/timeout cluster with no persisted fallback at all. Task #38 (pi-parity fix): `serve` gained
    /// its own `--idle-timeout-ms` flag (`ServeConfig::idle_timeout_ms` in `serve.rs`) falling back to
    /// this same setting, so both `run` and `serve` consult it now.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_provider_timeout_ms: Option<u64>,
    /// Used when neither `--retry-max-backoff-ms` nor `AI_AGENT_RETRY_MAX_BACKOFF_MS` is given — Task
    /// #30 (pi-parity feature): the retry/timeout cluster's third knob, `agent_core::client::
    /// GatewayClient::with_max_backoff` (`agent_core::client::MAX_BACKOFF`'s own 60s built-in ceiling),
    /// previously had no CLI flag or persisted override at all, unlike its two siblings
    /// (`default_retry_max_retries`/`default_retry_base_delay_ms` above). `None` leaves
    /// `MAX_BACKOFF`/`with_retry`'s own default ceiling in effect, unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_retry_max_backoff_ms: Option<u64>,
    /// Used when neither `--models` nor `AI_AGENT_MODELS` is given — pi-parity fix (Round 3): `serve`'s
    /// `cycle_model` candidate-scoping list previously had no persisted default, unlike `default_model`
    /// itself. An empty/absent explicit `--models` falls back to this list *in full*; a non-empty
    /// explicit `--models` wins outright with no merge between the two — matching every other field's
    /// plain override precedence (see [`Settings::merge_over`]'s own doc comment for why the project-tier
    /// merge treats this list field identically: replace, never append/concat). `run` has no `--models`
    /// flag of its own (nothing to scope — it never cycles models mid-run), so this is `serve`-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_models_list: Option<Vec<String>>,
    /// Used when no `--skill <path>`/`AI_AGENT_SKILL_PATH` is given, for both `run` and `serve` —
    /// pi-parity fix (Round 3): extra skill-discovery roots (beyond the two standard
    /// `~/.claude/skills`/`<cwd>/.claude/skills`) had no persisted default. An empty/absent explicit
    /// `--skill` list falls back to this list *in full*; a non-empty explicit list wins outright with no
    /// merge — same plain-override precedence as every other field (see
    /// [`Self::default_models_list`]'s doc comment).
    ///
    /// Since Round 20 (pi-parity feature), each entry is either a plain path — a directory (walked
    /// recursively, unchanged) or an individual `SKILL.md`/loose-`.md` file (already supported: this list
    /// feeds `skills::discover_with_diagnostics`'s own `extra_roots`, which has accepted a standalone file
    /// since the M3/pi-parity single-file fix) — or an override *pattern*, classified by
    /// [`classify_path_entry`] and applied uniformly across *every* discovery root (both standard ones and
    /// any plain path configured here), not just this list's own plain entries: `!pattern`/a bare entry
    /// containing `*`/`?` excludes an already-discovered skill matching by name, containing directory, or
    /// path; `+pattern` force-includes one an exclude elsewhere would otherwise have dropped; `-pattern`
    /// force-excludes one even if a `+pattern` elsewhere would have restored it. See
    /// [`crate::skills::apply_path_overrides`] for the application; matches pi's own
    /// `package-manager.ts`'s `isEnabledByOverrides`/`applyPatterns`, simplified to one uniform pass
    /// instead of pi's own tier-scoped (global vs. project `baseDir`) split — see `classify_path_entry`'s
    /// own doc comment for why.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_skill_paths: Option<Vec<String>>,
    /// Used when no `--prompt-template <path>`/`AI_AGENT_PROMPT_TEMPLATE_PATH` is given — pi-parity fix
    /// (Round 3), [`Self::default_skill_paths`]'s identical relationship to prompt-template discovery
    /// roots — including, since Round 20, the identical override-pattern/individual-file semantics
    /// described on that field's own doc comment (applied via [`crate::prompts::discover_with_diagnostics`]
    /// reusing the same [`crate::skills::apply_path_overrides`]/[`classify_path_entry`] this crate shares
    /// between skills and prompts everywhere else — e.g. `parse_frontmatter`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_prompt_template_paths: Option<Vec<String>>,
    /// Used when neither `--branch-summary-reserve-tokens` nor
    /// `AI_AGENT_BRANCH_SUMMARY_RESERVE_TOKENS` is given — Task #31 (pi-parity feature):
    /// `agent_core::Agent::with_branch_summary_reserve_tokens`/`branch_summary_reserve_tokens` existed
    /// with no caller anywhere in either binary, so a branch summary's own reserve budget was always
    /// hard-tied to whatever `compaction.reserve_tokens` resolved to, with no way to configure it
    /// independently — unlike pi's own `branchSummary.reserveTokens` (default 16384), which is a
    /// wholly separate setting from ordinary compaction's reserve. `None` leaves that hard-tied
    /// behavior in effect, unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_branch_summary_reserve_tokens: Option<u32>,
    /// A persisted, cross-session override for the mid-run *steer* lane's drain granularity — pi's own
    /// persisted `steeringMode` (pi-parity gap, pass 19): `agent_core::steering::QueueMode::{OneAtATime,
    /// All}` already existed and was already reachable at runtime via `serve`'s `set_steering_mode` RPC
    /// command, but nothing durable backed it — an operator wanting a standing default had to re-issue
    /// that RPC call at the start of every `serve` session. One of `"one_at_a_time"`/`"all"` (the exact
    /// wire vocabulary `set_steering_mode`'s own `mode` argument already uses — see `serve.rs`'s module
    /// doc comment), stored as a plain string rather than the enum itself, matching
    /// [`Self::default_reasoning_effort`]'s own "no hard dependency on `agent_core`'s exact type layout"
    /// convention. `None` defaults to `one_at_a_time`, matching both this crate's and pi's own default.
    /// Mirrors [`Self::compaction_enabled`]'s identical persist-on-RPC-toggle model rather than
    /// `default_reasoning_effort`'s CLI-mutated one: like auto-compaction, this only ever changes for the
    /// life of a running `serve` session (there's no `run`/`serve` one-shot analogue to a live toggle), so
    /// `serve`'s own `set_steering_mode` handler is expected to persist here itself (the way
    /// `set_auto_compaction` already calls [`SettingsStore::set_compaction_enabled`]), not a dedicated
    /// `agent settings` CLI flag. `run`'s and `serve`'s own `--steering-mode`/`AI_AGENT_STEERING_MODE`
    /// flags read this as their own stored-default fallback, same as any other persisted setting here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steering_mode: Option<String>,
    /// Same idea as [`Self::steering_mode`], for the follow-up lane drained at a stop boundary (plus any
    /// stranded steer messages swept in there) — matches pi's own separate `followUpMode`. Persisted (or
    /// not) independently of `steering_mode`, via `serve`'s own `set_follow_up_mode` RPC command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow_up_mode: Option<String>,
}

impl Settings {
    /// Deep-merge a `project`-tier [`Settings`] over `self` (the global tier), field by field: any
    /// `Some` on `project` overrides the corresponding field on `self`; a `None` field on `project` falls
    /// through to `self`'s own value unchanged. Mirrors pi's own two-tier design
    /// (`packages/coding-agent/src/core/settings-manager.ts`'s `deepMergeSettings`) — matched exactly for
    /// the `Vec<String>` list fields here (`default_models_list`/`default_skill_paths`/
    /// `default_prompt_template_paths`): pi's own array handling explicitly diverts any array-valued
    /// field out of its one-level-deep object-spread-merge branch (`!Array.isArray(...)` guards) into
    /// wholesale replacement, never a concat/append — so a project's list here, when set at all,
    /// *replaces* the global list outright rather than appending to it, matching that upstream behavior.
    /// This is also simpler and more predictable on its own merits: a project author reading their own
    /// `.claude/settings.json` sees exactly the list that applies, with no need to cross-reference the
    /// global file to reconstruct a merged result.
    ///
    /// Used by [`effective_settings_for_cwd`]'s project-tier resolution — never by [`SettingsStore`]
    /// itself, which has no project-tier concept of its own (see its own doc comment).
    #[must_use]
    pub fn merge_over(&self, project: &Settings) -> Settings {
        Settings {
            default_model: project
                .default_model
                .clone()
                .or_else(|| self.default_model.clone()),
            default_gateway_url: project
                .default_gateway_url
                .clone()
                .or_else(|| self.default_gateway_url.clone()),
            default_session_dir: project
                .default_session_dir
                .clone()
                .or_else(|| self.default_session_dir.clone()),
            default_project_trust: project.default_project_trust.or(self.default_project_trust),
            compaction_enabled: project.compaction_enabled.or(self.compaction_enabled),
            default_reasoning_effort: project
                .default_reasoning_effort
                .clone()
                .or_else(|| self.default_reasoning_effort.clone()),
            block_images: project.block_images.or(self.block_images),
            image_auto_resize: project.image_auto_resize.or(self.image_auto_resize),
            thinking_budget_overrides: project
                .thinking_budget_overrides
                .clone()
                .or_else(|| self.thinking_budget_overrides.clone()),
            default_bash_shell_path: project
                .default_bash_shell_path
                .clone()
                .or_else(|| self.default_bash_shell_path.clone()),
            default_bash_command_prefix: project
                .default_bash_command_prefix
                .clone()
                .or_else(|| self.default_bash_command_prefix.clone()),
            default_compaction_reserve_tokens: project
                .default_compaction_reserve_tokens
                .or(self.default_compaction_reserve_tokens),
            default_compaction_keep_recent_tokens: project
                .default_compaction_keep_recent_tokens
                .or(self.default_compaction_keep_recent_tokens),
            default_retry_max_retries: project
                .default_retry_max_retries
                .or(self.default_retry_max_retries),
            default_retry_base_delay_ms: project
                .default_retry_base_delay_ms
                .or(self.default_retry_base_delay_ms),
            default_provider_timeout_ms: project
                .default_provider_timeout_ms
                .or(self.default_provider_timeout_ms),
            default_retry_max_backoff_ms: project
                .default_retry_max_backoff_ms
                .or(self.default_retry_max_backoff_ms),
            default_models_list: project
                .default_models_list
                .clone()
                .or_else(|| self.default_models_list.clone()),
            default_skill_paths: project
                .default_skill_paths
                .clone()
                .or_else(|| self.default_skill_paths.clone()),
            default_prompt_template_paths: project
                .default_prompt_template_paths
                .clone()
                .or_else(|| self.default_prompt_template_paths.clone()),
            default_branch_summary_reserve_tokens: project
                .default_branch_summary_reserve_tokens
                .or(self.default_branch_summary_reserve_tokens),
            steering_mode: project.steering_mode.clone().or_else(|| self.steering_mode.clone()),
            follow_up_mode: project
                .follow_up_mode
                .clone()
                .or_else(|| self.follow_up_mode.clone()),
        }
    }
}

/// One classified entry from [`Settings::default_skill_paths`]/[`Settings::default_prompt_template_paths`]
/// (equivalently, an already-merged `extra_roots` list — an operator-supplied `--skill`/
/// `--prompt-template` path falls through the same list, see `main.rs`'s `extra_skill_paths`/
/// `extra_prompt_template_paths` resolution) — pi-parity feature, Round 20: pi's own
/// `package-manager.ts` treats skills/prompts as configurable "resources" whose discovery-root list can
/// also carry override patterns (`isPattern`/`isOverridePattern`/`splitPatterns`), not just plain
/// directory paths; beyond had no such layer at all before this.
///
/// Deliberately still a plain `Vec<String>` on `Settings` itself (see [`Settings::merge_over`]'s doc
/// comment on why every list field there is wholesale-replaced, never appended, between tiers) — this
/// type only exists to classify one already-resolved entry once `skills::discover_with_diagnostics`/
/// `prompts::discover_with_diagnostics` read it, not to change what's actually persisted on disk.
///
/// Pi's own equivalent is a dual-purpose split: a bare, unprefixed glob is an *include* filter (`applyPatterns`'s
/// `includes` bucket — "restrict to only files matching one of these"), scoped only to the files an
/// explicit plain directory entry *in that same settings list* turned up, while `!`/`+`/`-`-prefixed
/// entries are separately read as *overrides* (`getOverridePatterns`/`isEnabledByOverrides`) applied only
/// to each settings tier's own fixed auto-discovery directory (global entries only ever gate the global
/// standard root, project entries only the project one). Beyond deliberately simplifies both of those
/// splits into one uniform pass — see [`crate::skills::apply_path_overrides`], which this variant's
/// caller applies identically to *every* discovery root (standard and configured alike) after beyond's
/// own [`Settings::merge_over`] has already collapsed global+project into one effective list, so there is
/// no separate tier/`baseDir` left to scope patterns to by the time discovery runs. Concretely: a bare
/// glob here is folded into [`PathEntry::Exclude`] rather than kept as a separate "only these" whitelist
/// bucket — pi's own whitelist behavior only ever does anything when combined with an explicit directory
/// entry in the very same list (with no such entry, `applyPatterns`'s own `allFiles` is empty and the
/// bare glob filters nothing at all), a narrower and more surprising rule than "a bare glob excludes
/// whatever it matches, everywhere," which is what an operator skimming a `skills: ["!draft-*"]` entry
/// would reasonably expect it to mean.
pub(crate) enum PathEntry<'a> {
    /// A literal directory or individual file path — an additional discovery root, exactly what every
    /// entry in this list meant before Round 20.
    Root(&'a str),
    /// `!pattern`, or a bare entry containing a glob metacharacter (`*`/`?`) with no `+`/`-` prefix —
    /// exclude any already-discovered skill/prompt whose name, containing directory (for a
    /// directory-shaped skill), or path matches.
    Exclude(&'a str),
    /// `+pattern` — force-include a skill/prompt an [`PathEntry::Exclude`] elsewhere in the same list
    /// would otherwise have dropped.
    Include(&'a str),
    /// `-pattern` — force-exclude a skill/prompt even if a [`PathEntry::Include`] elsewhere in the same
    /// list would have restored it.
    ForceExclude(&'a str),
}

/// Classify one raw `default_skill_paths`/`default_prompt_template_paths` entry — see [`PathEntry`] for
/// the exact semantics each variant carries. Prefix stripping happens here, once, so every caller (the
/// root-walk loop and [`crate::skills::apply_path_overrides`] alike) agrees on the same four buckets
/// rather than re-deriving them; matches pi's own `isOverridePattern`/`hasGlobPattern` prefix rules
/// exactly (`!`/`+`/`-` always wins as an override regardless of whether the remainder also happens to
/// contain a glob metacharacter — a literal `!my-skill` with no `*`/`?` at all is still `Exclude`, not a
/// literal-path `Root`).
pub(crate) fn classify_path_entry(entry: &str) -> PathEntry<'_> {
    if let Some(rest) = entry.strip_prefix('!') {
        PathEntry::Exclude(rest)
    } else if let Some(rest) = entry.strip_prefix('+') {
        PathEntry::Include(rest)
    } else if let Some(rest) = entry.strip_prefix('-') {
        PathEntry::ForceExclude(rest)
    } else if entry.contains('*') || entry.contains('?') {
        PathEntry::Exclude(entry)
    } else {
        PathEntry::Root(entry)
    }
}

/// A persisted settings file, one per machine — the *global* tier. A project-level tier also exists now
/// (Round 3, pi-parity feature: [`effective_settings_for_cwd`], `<cwd>/.claude/settings.json`, deep-merged
/// on top of this one via [`Settings::merge_over`] whenever `cwd` is trusted), matching pi's own two-tier
/// `settings.json` design — but this type itself remains the global tier only. `agent settings` (the CLI
/// subcommand) manages only this global tier; a project's own `.claude/settings.json` is hand-edited,
/// like `models.json`, with no CLI mutation surface of its own.
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

    /// Set (`Some`) or clear (`None`) the persisted `--no-image-auto-resize` default, persisting
    /// atomically.
    pub fn set_image_auto_resize(&mut self, image_auto_resize: Option<bool>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.image_auto_resize = image_auto_resize)
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

    /// Set (`Some`) or clear (`None`) the stored default `--bash-shell-path`, persisting atomically.
    pub fn set_default_bash_shell_path(&mut self, path: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_bash_shell_path = path)
    }

    /// Set (`Some`) or clear (`None`) the stored default `--bash-command-prefix`, persisting atomically.
    pub fn set_default_bash_command_prefix(&mut self, prefix: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_bash_command_prefix = prefix)
    }

    /// Set (`Some`) or clear (`None`) the stored default compaction reserve-token override, persisting
    /// atomically.
    pub fn set_default_compaction_reserve_tokens(&mut self, tokens: Option<u32>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_compaction_reserve_tokens = tokens)
    }

    /// Set (`Some`) or clear (`None`) the stored default compaction keep-recent-token override,
    /// persisting atomically.
    pub fn set_default_compaction_keep_recent_tokens(
        &mut self,
        tokens: Option<u32>,
    ) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_compaction_keep_recent_tokens = tokens)
    }

    /// Set (`Some`) or clear (`None`) the stored default retry max-retries override, persisting
    /// atomically.
    pub fn set_default_retry_max_retries(&mut self, retries: Option<u32>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_retry_max_retries = retries)
    }

    /// Set (`Some`) or clear (`None`) the stored default retry base-delay override (milliseconds),
    /// persisting atomically.
    pub fn set_default_retry_base_delay_ms(&mut self, ms: Option<u64>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_retry_base_delay_ms = ms)
    }

    /// Set (`Some`) or clear (`None`) the stored default provider (idle-read) timeout override
    /// (milliseconds), persisting atomically.
    pub fn set_default_provider_timeout_ms(&mut self, ms: Option<u64>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_provider_timeout_ms = ms)
    }

    /// Set (`Some`) or clear (`None`) the stored default retry backoff-ceiling override
    /// (milliseconds), persisting atomically.
    pub fn set_default_retry_max_backoff_ms(&mut self, ms: Option<u64>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_retry_max_backoff_ms = ms)
    }

    /// Set (`Some`) or clear (`None`) the stored default `--models` scoping/cycling list, persisting
    /// atomically.
    pub fn set_default_models_list(&mut self, models: Option<Vec<String>>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_models_list = models)
    }

    /// Set (`Some`) or clear (`None`) the stored default extra skill-discovery paths, persisting
    /// atomically.
    pub fn set_default_skill_paths(&mut self, paths: Option<Vec<String>>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_skill_paths = paths)
    }

    /// Set (`Some`) or clear (`None`) the stored default extra prompt-template-discovery paths,
    /// persisting atomically.
    pub fn set_default_prompt_template_paths(
        &mut self,
        paths: Option<Vec<String>>,
    ) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_prompt_template_paths = paths)
    }

    /// Set (`Some`) or clear (`None`) the stored default branch-summary reserve-token override,
    /// persisting atomically.
    pub fn set_default_branch_summary_reserve_tokens(
        &mut self,
        tokens: Option<u32>,
    ) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.default_branch_summary_reserve_tokens = tokens)
    }

    /// Set (`Some`) or clear (`None`) the persisted steer-lane `QueueMode` default, persisting
    /// atomically. Takes the already-validated wire string (`"one_at_a_time"`/`"all"`) — mirrors
    /// `set_default_reasoning_effort`'s identical shape (a plain string, validated by the caller, whether
    /// that's `main.rs`'s CLI parsing or `serve`'s own `set_steering_mode` RPC handler, before it ever
    /// reaches here).
    pub fn set_steering_mode(&mut self, mode: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.steering_mode = mode)
    }

    /// Set (`Some`) or clear (`None`) the persisted follow-up-lane `QueueMode` default, persisting
    /// atomically. [`Self::set_steering_mode`]'s identical shape/rationale, for `serve`'s
    /// `set_follow_up_mode` RPC command.
    pub fn set_follow_up_mode(&mut self, mode: Option<String>) -> std::io::Result<()> {
        self.mutate_locked(move |s| s.follow_up_mode = mode)
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
/// state, gracefully defaulted." A missing file is the ordinary case (no settings configured yet) and
/// stays silent; an existing-but-unparsable file (corrupted, hand-edited into invalid JSON, or written
/// by a newer binary with a field this one doesn't understand) `warn!`s before defaulting — mirrors
/// [`ModelOverrides::open`]'s identical "skip and warn, don't silently lose data" convention, which this
/// function used to be the one exception to: silently resetting to all-unset with no diagnostic at all.
/// That silence was especially costly here, since [`SettingsStore::mutate_locked`] re-reads through
/// this same function and then *writes back* whatever it returns — a single transient parse failure
/// during any `agent settings set-*` call could permanently overwrite every previously-configured field
/// with no warning before or after.
fn read_store_file(path: &Path) -> Settings {
    let contents = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == ErrorKind::NotFound => return Settings::default(),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "could not read settings file, treating it as empty"
            );
            return Settings::default();
        }
    };
    match serde_json::from_str(&contents) {
        Ok(settings) => settings,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "settings file is not valid JSON, treating it as empty"
            );
            Settings::default()
        }
    }
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

/// Where a project-level settings tier lives, if trusted: `<cwd>/.claude/settings.json` — a sibling of
/// the same project's `.claude/SYSTEM.md`/`skills`/`prompts` (see
/// `trust_store::has_trust_gated_resources`), gated by the exact same persisted-trust mechanism (Round 3,
/// pi-parity feature).
pub fn project_settings_path(cwd: &Path) -> PathBuf {
    cwd.join(".claude/settings.json")
}

/// The effective settings for a `run`/`serve` invocation in `cwd`: the global tier
/// ([`SettingsStore::open_default`]) with a project-level `<cwd>/.claude/settings.json` tier deep-merged
/// on top ([`Settings::merge_over`]) — but *only* when `cwd` is trusted per the persisted
/// [`crate::trust_store::TrustStore`] allowlist (`TrustStore::is_trusted`: the exact-path-or-inherited
/// grant `agent trust`/`agent untrust` manage). An untrusted project's `.claude/settings.json` is
/// completely ignored — not partially applied — since a hostile untrusted checkout could otherwise use it
/// to redirect `default_gateway_url`/`default_model` (or any other field) to something
/// attacker-controlled just by shipping the file; the global tier alone is returned in that case,
/// identical to `SettingsStore::open_default().get().clone()`.
///
/// Deliberately checks only the persisted allowlist, not the fuller `resolve_project_trust` precedence
/// `run`/`serve` apply for `SYSTEM.md`/skills/prompts (which also honors a one-shot `--trust-project`
/// flag and a "nothing here to gate" fast path) — a transient, session-scoped `--trust-project` is an
/// operator saying "honor this project's *identity* override for this one run", not "also let this
/// project's settings.json silently redirect where my credentials/requests go." Only a *persisted*
/// `agent trust <path>` grant is trusted enough for that.
///
/// Returns a plain [`Settings`] snapshot, not a [`SettingsStore`] handle: there's no project-tier
/// mutation surface analogous to `agent settings` for the global tier — a project's
/// `.claude/settings.json` is hand-edited (matching `models.json`'s own convention), not managed via any
/// CLI flag here.
pub fn effective_settings_for_cwd(cwd: &Path) -> Settings {
    effective_settings_with_stores(
        &SettingsStore::open_default(),
        cwd,
        &crate::trust_store::TrustStore::open_default(),
    )
}

/// [`effective_settings_for_cwd`], but against already-open stores — split out so tests can point both
/// at temporary paths instead of the real `~/.claude` (see `ModelOverride::resolved_api_key_with_env`'s
/// doc comment for why this crate prefers threading test doubles through rather than mutating real,
/// process-wide state like `$HOME`).
fn effective_settings_with_stores(
    global: &SettingsStore,
    cwd: &Path,
    trust: &crate::trust_store::TrustStore,
) -> Settings {
    if !trust.is_trusted(cwd) {
        return global.get().clone();
    }
    let project = read_store_file(&project_settings_path(cwd));
    global.get().merge_over(&project)
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
    /// `agent_core::client::DirectRouting::auth_header` in `gateway_credential::resolve_gateway_credential`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<String>,
    /// Prepended to `api_key`'s resolved value when sent through [`Self::auth_header`] — pi-parity Fix
    /// 4 (Round 2): Cloudflare AI Gateway wants `cf-aig-authorization: Bearer <key>`, a *named* header
    /// (already covered by `auth_header` alone) carrying a Bearer-*prefixed* value (which bare
    /// `auth_header` doesn't — it sends the credential verbatim, correct for Azure's `api-key` scheme
    /// but wrong here). `None` (the default) keeps `auth_header`'s existing bare-value behavior. Has
    /// no effect without `auth_header` also set — there's no header to prefix a value into otherwise.
    /// Threaded to `agent_core::client::DirectRouting::auth_header_prefix`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header_prefix: Option<String>,
    /// Override which provider wire (Anthropic/OpenAI Chat Completions/OpenAI Responses) this model's
    /// requests build and decode as, instead of `agent_core::dialect::Dialect::for_model`'s own
    /// name-heuristic (a "claude"/"anthropic" substring ⇒ Anthropic, else OpenAI-family) — pi-parity
    /// Fix 1 (Round 2). Only meaningful alongside [`Self::base_url`]: a plain gateway-relayed
    /// request's provider routing is inherently tied to which dialect-specific endpoint path the
    /// gateway's own known-provider table matches on, so an isolated `dialect` override with no
    /// companion `base_url` would have nothing to attach to. The motivating case is a genuinely
    /// Anthropic-wire third-party provider whose model ids don't carry a "claude"/"anthropic"
    /// substring (Kimi-Coding's `kimi-k2-thinking`, reached via `api.kimi.com/coding`) — without this,
    /// beyond builds an OpenAI-shaped body and POSTs it to an Anthropic Messages endpoint, a hard
    /// wire-format mismatch. One of `"anthropic"`/`"openai"`/`"openai_responses"` (see
    /// [`agent_core::dialect::Dialect`]'s own `Serialize`/`Deserialize` impl). `None` (the default)
    /// keeps the existing heuristic. Threaded to `agent_core::client::DirectRouting::dialect_override`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialect: Option<agent_core::dialect::Dialect>,
    /// For a *direct* (`base_url`-set) override: send this instead of the app-level model id (the
    /// string passed to `--model`/configured here as this override's own key) as the request body's
    /// wire-level `"model"` field — pi-parity Fix 2 (Round 2): Azure OpenAI's deployment name doesn't
    /// have to match the model id used for capability lookups (mirrors pi's own
    /// `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` env var in `azure-openai-responses.ts`, expressed here as a
    /// per-model-id `models.json` field instead of a separate comma-joined map, since this override is
    /// already keyed per model id). `None` (the default) leaves the wire-level `"model"` field as the
    /// app-level id. Threaded to `agent_core::client::DirectRouting::deployment_name`.
    ///
    /// Also inserted as a URL path segment — pi-parity Fix 3 (Round 2, Task 46): Azure's *classic*
    /// dated-`api-version` REST convention addresses the deployment purely by URL
    /// (`/openai/deployments/{name}/chat/completions?api-version=…`), never through the body's
    /// `"model"` field at all (that's the *other*, newer unified-`/openai/v1` convention's own
    /// signal). `gateway_credential::resolve_gateway_credential`'s `direct_route_base_and_path` folds
    /// this in alongside the body substitution above whenever both this field and [`Self::base_url`]
    /// are set, so the same field covers both real Azure REST shapes rather than needing a second,
    /// URL-specific flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment_name: Option<String>,
    /// For a *direct* (`base_url`-set) override: Azure's dated REST `api-version` (e.g.
    /// `"2024-08-01-preview"`), sent as a `?api-version=…` query-string parameter on every request —
    /// pi-parity Fix 2 (Round 2): most real enterprise Azure OpenAI deployments are pinned to a dated
    /// API version (pi's own default, absent an explicit override, is `"v1"` —
    /// `DEFAULT_AZURE_API_VERSION` in `azure-openai-responses.ts`), and beyond previously had no way to
    /// send one at all. `None` (the default) sends no query string, unchanged — deliberately no
    /// auto-detection from `base_url` looking like an Azure host: an explicit field is simpler to
    /// reason about than a heuristic that silently changes behavior based on hostname shape. Threaded
    /// to `agent_core::client::DirectRouting::query` (percent-encoded at resolution time, since this is
    /// a bare value here, not a pre-built query string).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
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
    /// fails to resolve (an unset env var, a failing `!command`) or resolves to an empty string (a `$VAR`
    /// set but empty, or a literal `""`) rather than sending a literal `$VAR`/`!command` string, or an
    /// empty value, as a header — same empty-string guard as
    /// [`Self::resolved_api_key_with_env`], just dropping the entry outright instead of falling back
    /// (there's no per-header fallback value to fall back to).
    ///
    /// Seeded first with [`default_headers_for_base_url`]'s provider-mandatory defaults (Fixes #37/#38,
    /// pi-parity audit — NVIDIA's `NVCF-POLL-SECONDS`, Kimi-Coding's `User-Agent`), which
    /// `self.headers`'s own resolved entries are then layered on top of: a user-supplied value for the
    /// same header name always wins over the seeded default, never the other way around.
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
        let mut resolved: std::collections::HashMap<String, String> = self
            .base_url
            .as_deref()
            .map(default_headers_for_base_url)
            .unwrap_or(&[])
            .iter()
            .map(|&(name, value)| (name.to_string(), value.to_string()))
            .collect();
        resolved.extend(self.headers.iter().filter_map(|(k, v)| {
            resolve_config_value(v, env_overrides)
                .filter(|v| !v.is_empty())
                .map(|v| (k.clone(), v))
        }));
        resolved
    }
}

/// Provider-mandatory default headers seeded automatically for a BYO [`ModelOverride::base_url`]
/// pointing at a known host that requires them at the HTTP layer — headers a `dialect` override alone
/// (which only fixes the wire *body* shape) can't cover. Additive with, and always overridden by, any
/// user-supplied [`ModelOverride::headers`] entry for the same name (see
/// [`ModelOverride::resolved_headers_with_env`]) — this only fills a gap the operator would otherwise
/// have to independently discover and hand-author from the provider's own docs.
///
/// - NVIDIA Cloud Functions-backed endpoints (Fix #37, pi-parity audit): `integrate.api.nvidia.com`
///   (pi's own `nvidia.models.ts` `baseUrl`) — NIM-backed endpoints can answer a request with an async
///   202 + poll URL instead of a synchronous stream; `NVCF-POLL-SECONDS` tells the server to long-poll
///   instead, mirroring pi's own catalog, which sets this header unconditionally on every entry.
/// - Kimi-Coding (Fix #38, pi-parity audit): `api.kimi.com/coding` (pi's own `kimi-coding.models.ts`
///   `baseUrl`) is an official-CLI-only endpoint that appears to gate on `User-Agent`; pi's own catalog
///   sets this unconditionally on every entry too. A prior pass's `ModelOverride::dialect` fix only
///   addressed the wire *body* shape (Anthropic vs. OpenAI) — without this header, requests may still
///   fail at the HTTP layer even with a correctly-shaped body.
///
/// Matched by a plain substring check against `base_url` (not a parsed-host equality check, unlike
/// `gateway_credential::is_azure_host`): both hosts here are always used with a fixed, single-segment
/// path baked into pi's own catalog (`/v1`, `/coding`), so there's no meaningfully different shape an
/// operator's own `base_url` could take that would make host-parsing worth the extra complexity.
fn default_headers_for_base_url(base_url: &str) -> &'static [(&'static str, &'static str)] {
    if base_url.contains("integrate.api.nvidia.com") {
        &[("NVCF-POLL-SECONDS", "3600")]
    } else if base_url.contains("api.kimi.com") {
        &[("User-Agent", "KimiCLI/1.5")]
    } else {
        &[]
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
    fn a_corrupt_settings_file_warns_and_is_treated_as_empty() {
        // Regression: `read_store_file` used to silently reset to all-unset on any read/parse
        // failure, with zero diagnostic — unlike `ModelOverrides::open`'s identical case a few tests
        // up, which has always warned. Worse, `mutate_locked` re-reads through this same function and
        // writes back whatever it returns, so a transient parse failure during any `agent settings
        // set-*` call could permanently overwrite every previously-configured field with no warning
        // before or after.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(&path, "not valid json at all { [ }").unwrap();

        let capture = crate::tracing_test::capture(|| {
            let store = SettingsStore::open(path);
            assert_eq!(store.get(), &Settings::default());
        });
        assert!(
            capture.messages().iter().any(|m| m.contains("settings")),
            "a corrupt settings file must be logged, not silently swallowed: {:?}",
            capture.messages()
        );
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
    fn set_steering_mode_and_set_follow_up_mode_persist_independently_and_reopening_sees_both() {
        // Task 1 (pi-parity fix, pass 19): `steering_mode`/`follow_up_mode` are a new persisted pair,
        // each independently settable/clearable — mirrors `set_compaction_enabled`'s identical
        // persist-on-RPC-toggle round-trip above, but for two sibling fields at once, proving neither
        // setter clobbers the other's stored value.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().steering_mode, None, "no stored override by default");
        assert_eq!(store.get().follow_up_mode, None, "no stored override by default");

        store.set_steering_mode(Some("all".to_string())).unwrap();
        assert_eq!(store.get().steering_mode.as_deref(), Some("all"));
        assert_eq!(
            store.get().follow_up_mode,
            None,
            "setting steering_mode must not touch follow_up_mode"
        );
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().steering_mode.as_deref(), Some("all"));

        store
            .set_follow_up_mode(Some("one_at_a_time".to_string()))
            .unwrap();
        assert_eq!(store.get().follow_up_mode.as_deref(), Some("one_at_a_time"));
        assert_eq!(
            store.get().steering_mode.as_deref(),
            Some("all"),
            "setting follow_up_mode must not clobber the already-stored steering_mode"
        );

        // Clearing one back to `None` must also round-trip, independent of the other.
        store.set_steering_mode(None).unwrap();
        assert_eq!(store.get().steering_mode, None);
        assert_eq!(store.get().follow_up_mode.as_deref(), Some("one_at_a_time"));
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().steering_mode, None);
        assert_eq!(reopened.get().follow_up_mode.as_deref(), Some("one_at_a_time"));
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
    fn set_image_auto_resize_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().image_auto_resize, None);

        store.set_image_auto_resize(Some(false)).unwrap();
        assert_eq!(store.get().image_auto_resize, Some(false));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().image_auto_resize, Some(false));

        let mut store = SettingsStore::open(path.clone());
        store.set_image_auto_resize(None).unwrap();
        assert_eq!(store.get().image_auto_resize, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().image_auto_resize, None);
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
    fn resolved_headers_drops_an_entry_that_resolves_to_an_empty_string() {
        // Fix 4 (pi-parity gap): a header value resolving to `""` — e.g. a `$VAR` reference that's set
        // but empty — must be dropped like an unresolvable one, not sent as a literal empty-value
        // header. Mirrors `resolved_api_key_with_env`'s identical `.filter(|v| !v.is_empty())` guard.
        let mut over = ModelOverride::default();
        over.headers
            .insert("X-Empty".to_string(), "$EMPTY_VAR".to_string());
        over.headers
            .insert("X-Present".to_string(), "literal".to_string());
        let mut env = std::collections::HashMap::new();
        env.insert("EMPTY_VAR".to_string(), String::new());

        let resolved = over.resolved_headers_with_env(Some(&env));
        assert_eq!(resolved.get("X-Empty"), None, "got: {resolved:?}");
        assert_eq!(resolved.get("X-Present").map(String::as_str), Some("literal"));
    }

    // Fixes #37/#38 (pi-parity audit): a BYO `base_url` pointed at a known provider host that requires
    // a mandatory HTTP-layer header gets it seeded automatically, without the operator having to
    // discover and hand-author it themselves.

    #[test]
    fn resolved_headers_seeds_the_nvidia_poll_seconds_header_for_the_nvidia_hosted_base_url() {
        let over = ModelOverride {
            base_url: Some("https://integrate.api.nvidia.com/v1".to_string()),
            ..Default::default()
        };
        assert_eq!(
            over.resolved_headers().get("NVCF-POLL-SECONDS").map(String::as_str),
            Some("3600")
        );
    }

    #[test]
    fn resolved_headers_seeds_the_kimi_coding_user_agent_header_for_the_kimi_hosted_base_url() {
        let over = ModelOverride {
            base_url: Some("https://api.kimi.com/coding".to_string()),
            ..Default::default()
        };
        assert_eq!(
            over.resolved_headers().get("User-Agent").map(String::as_str),
            Some("KimiCLI/1.5")
        );
    }

    #[test]
    fn resolved_headers_seeds_no_default_header_for_an_unrelated_base_url() {
        let over = ModelOverride {
            base_url: Some("https://my-ollama-box.example.com".to_string()),
            ..Default::default()
        };
        assert!(over.resolved_headers().is_empty());
    }

    #[test]
    fn resolved_headers_seeds_no_default_header_when_base_url_is_unset() {
        let over = ModelOverride {
            headers: {
                let mut h = std::collections::BTreeMap::new();
                h.insert("X-Custom".to_string(), "value".to_string());
                h
            },
            ..Default::default()
        };
        let resolved = over.resolved_headers();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved.get("X-Custom").map(String::as_str), Some("value"));
    }

    #[test]
    fn resolved_headers_a_user_supplied_value_overrides_the_seeded_nvidia_default() {
        // Additive, never overriding, per this fn's own doc comment: an operator explicitly setting
        // `headers` for the same name must win over the auto-seeded default.
        let mut over = ModelOverride {
            base_url: Some("https://integrate.api.nvidia.com/v1".to_string()),
            ..Default::default()
        };
        over.headers
            .insert("NVCF-POLL-SECONDS".to_string(), "60".to_string());
        assert_eq!(
            over.resolved_headers().get("NVCF-POLL-SECONDS").map(String::as_str),
            Some("60")
        );
    }

    #[test]
    fn resolved_headers_combines_the_seeded_default_with_other_user_supplied_headers() {
        let mut over = ModelOverride {
            base_url: Some("https://api.kimi.com/coding".to_string()),
            ..Default::default()
        };
        over.headers
            .insert("X-Extra".to_string(), "extra-value".to_string());
        let resolved = over.resolved_headers();
        assert_eq!(resolved.get("User-Agent").map(String::as_str), Some("KimiCLI/1.5"));
        assert_eq!(resolved.get("X-Extra").map(String::as_str), Some("extra-value"));
        assert_eq!(resolved.len(), 2);
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
    fn dialect_override_round_trips_through_json_and_defaults_to_none() {
        // Fix 1 (pi-parity, Round 2): a `models.json` `dialect` field names one of
        // `agent_core::dialect::Dialect`'s own short JSON strings directly.
        let over = ModelOverride::default();
        assert_eq!(over.dialect, None);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        fs::write(
            &path,
            r#"{"kimi-k2-thinking": {"base_url": "https://api.kimi.com/coding", "dialect": "anthropic"}}"#,
        )
        .unwrap();

        let overrides = ModelOverrides::open(path);
        let over = overrides.get("kimi-k2-thinking").unwrap();
        assert_eq!(over.dialect, Some(agent_core::dialect::Dialect::Anthropic));
    }

    #[test]
    fn dialect_override_is_omitted_from_serialized_json_when_unset() {
        let over = ModelOverride {
            base_url: Some("https://example.com".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&over).unwrap();
        assert!(
            !json.contains("dialect"),
            "unset dialect must not appear in the serialized form, got: {json}"
        );
    }

    #[test]
    fn an_unrecognized_dialect_string_fails_to_parse_the_whole_file_which_is_treated_as_empty() {
        // Matches every other field's forward-compatibility posture (see
        // `unrecognized_fields_in_the_file_are_ignored_not_a_parse_error`): a *field value* that fails
        // to deserialize fails that model's whole entry, and `ModelOverrides::open` degrades the whole
        // file to empty (with a `warn!`) rather than panicking the process over a hand-edited typo.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        fs::write(
            &path,
            r#"{"bad-model": {"base_url": "https://example.com", "dialect": "not-a-real-dialect"}}"#,
        )
        .unwrap();

        let capture = crate::tracing_test::capture(|| {
            let overrides = ModelOverrides::open(path);
            assert!(overrides.is_empty());
        });
        assert!(
            capture
                .messages()
                .iter()
                .any(|m| m.contains("model overrides")),
            "got: {:?}",
            capture.messages()
        );
    }

    #[test]
    fn deployment_name_and_api_version_round_trip_through_json_and_default_to_none() {
        // Fix 2 (pi-parity, Round 2): Azure OpenAI's deployment-name mapping and dated `api-version`.
        let over = ModelOverride::default();
        assert_eq!(over.deployment_name, None);
        assert_eq!(over.api_version, None);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        fs::write(
            &path,
            r#"{"gpt-4o-azure": {"base_url": "https://foo.openai.azure.com/openai/v1", "deployment_name": "my-deployment", "api_version": "2024-08-01-preview"}}"#,
        )
        .unwrap();

        let overrides = ModelOverrides::open(path);
        let over = overrides.get("gpt-4o-azure").unwrap();
        assert_eq!(over.deployment_name.as_deref(), Some("my-deployment"));
        assert_eq!(over.api_version.as_deref(), Some("2024-08-01-preview"));
    }

    #[test]
    fn deployment_name_and_api_version_are_omitted_from_serialized_json_when_unset() {
        let over = ModelOverride {
            base_url: Some("https://example.com".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&over).unwrap();
        assert!(!json.contains("deployment_name"), "got: {json}");
        assert!(!json.contains("api_version"), "got: {json}");
    }

    #[test]
    fn auth_header_prefix_round_trips_through_json_and_defaults_to_none() {
        // Fix 4 (pi-parity, Round 2): Cloudflare AI Gateway's Bearer-prefixed named header.
        let over = ModelOverride::default();
        assert_eq!(over.auth_header_prefix, None);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.json");
        fs::write(
            &path,
            r#"{"cf-model": {"base_url": "https://gateway.ai.cloudflare.com/x", "auth_header": "cf-aig-authorization", "auth_header_prefix": "Bearer "}}"#,
        )
        .unwrap();

        let overrides = ModelOverrides::open(path);
        let over = overrides.get("cf-model").unwrap();
        assert_eq!(over.auth_header.as_deref(), Some("cf-aig-authorization"));
        assert_eq!(over.auth_header_prefix.as_deref(), Some("Bearer "));
    }

    #[test]
    fn auth_header_prefix_is_omitted_from_serialized_json_when_unset() {
        let over = ModelOverride {
            base_url: Some("https://example.com".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&over).unwrap();
        assert!(
            !json.contains("auth_header_prefix"),
            "unset auth_header_prefix must not appear in the serialized form, got: {json}"
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

    // --- Round 3 (pi-parity): five more flag/env-only settings gain a persisted stored-default fallback.

    #[test]
    fn set_default_bash_shell_path_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_bash_shell_path, None);

        store
            .set_default_bash_shell_path(Some("/bin/zsh".to_string()))
            .unwrap();
        assert_eq!(store.get().default_bash_shell_path.as_deref(), Some("/bin/zsh"));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(
            reopened.get().default_bash_shell_path.as_deref(),
            Some("/bin/zsh")
        );

        let mut store = SettingsStore::open(path.clone());
        store.set_default_bash_shell_path(None).unwrap();
        assert_eq!(store.get().default_bash_shell_path, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_bash_shell_path, None);
    }

    #[test]
    fn set_default_bash_command_prefix_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_bash_command_prefix, None);

        store
            .set_default_bash_command_prefix(Some("source .venv/bin/activate".to_string()))
            .unwrap();
        assert_eq!(
            store.get().default_bash_command_prefix.as_deref(),
            Some("source .venv/bin/activate")
        );
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(
            reopened.get().default_bash_command_prefix.as_deref(),
            Some("source .venv/bin/activate")
        );

        let mut store = SettingsStore::open(path.clone());
        store.set_default_bash_command_prefix(None).unwrap();
        assert_eq!(store.get().default_bash_command_prefix, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_bash_command_prefix, None);
    }

    #[test]
    fn set_default_compaction_reserve_tokens_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_compaction_reserve_tokens, None);

        store.set_default_compaction_reserve_tokens(Some(8_192)).unwrap();
        assert_eq!(store.get().default_compaction_reserve_tokens, Some(8_192));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().default_compaction_reserve_tokens, Some(8_192));

        let mut store = SettingsStore::open(path.clone());
        store.set_default_compaction_reserve_tokens(None).unwrap();
        assert_eq!(store.get().default_compaction_reserve_tokens, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_compaction_reserve_tokens, None);
    }

    #[test]
    fn set_default_compaction_keep_recent_tokens_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_compaction_keep_recent_tokens, None);

        store
            .set_default_compaction_keep_recent_tokens(Some(12_000))
            .unwrap();
        assert_eq!(
            store.get().default_compaction_keep_recent_tokens,
            Some(12_000)
        );
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(
            reopened.get().default_compaction_keep_recent_tokens,
            Some(12_000)
        );

        let mut store = SettingsStore::open(path.clone());
        store.set_default_compaction_keep_recent_tokens(None).unwrap();
        assert_eq!(store.get().default_compaction_keep_recent_tokens, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_compaction_keep_recent_tokens, None);
    }

    #[test]
    fn set_default_retry_max_retries_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_retry_max_retries, None);

        store.set_default_retry_max_retries(Some(5)).unwrap();
        assert_eq!(store.get().default_retry_max_retries, Some(5));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().default_retry_max_retries, Some(5));

        let mut store = SettingsStore::open(path.clone());
        store.set_default_retry_max_retries(None).unwrap();
        assert_eq!(store.get().default_retry_max_retries, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_retry_max_retries, None);
    }

    #[test]
    fn set_default_retry_base_delay_ms_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_retry_base_delay_ms, None);

        store.set_default_retry_base_delay_ms(Some(500)).unwrap();
        assert_eq!(store.get().default_retry_base_delay_ms, Some(500));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().default_retry_base_delay_ms, Some(500));

        let mut store = SettingsStore::open(path.clone());
        store.set_default_retry_base_delay_ms(None).unwrap();
        assert_eq!(store.get().default_retry_base_delay_ms, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_retry_base_delay_ms, None);
    }

    #[test]
    fn set_default_provider_timeout_ms_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_provider_timeout_ms, None);

        store.set_default_provider_timeout_ms(Some(60_000)).unwrap();
        assert_eq!(store.get().default_provider_timeout_ms, Some(60_000));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().default_provider_timeout_ms, Some(60_000));

        let mut store = SettingsStore::open(path.clone());
        store.set_default_provider_timeout_ms(None).unwrap();
        assert_eq!(store.get().default_provider_timeout_ms, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_provider_timeout_ms, None);
    }

    #[test]
    fn set_default_retry_max_backoff_ms_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_retry_max_backoff_ms, None);

        store.set_default_retry_max_backoff_ms(Some(30_000)).unwrap();
        assert_eq!(store.get().default_retry_max_backoff_ms, Some(30_000));
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(reopened.get().default_retry_max_backoff_ms, Some(30_000));

        let mut store = SettingsStore::open(path.clone());
        store.set_default_retry_max_backoff_ms(None).unwrap();
        assert_eq!(store.get().default_retry_max_backoff_ms, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_retry_max_backoff_ms, None);
    }

    #[test]
    fn set_default_branch_summary_reserve_tokens_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_branch_summary_reserve_tokens, None);

        store
            .set_default_branch_summary_reserve_tokens(Some(16_384))
            .unwrap();
        assert_eq!(
            store.get().default_branch_summary_reserve_tokens,
            Some(16_384)
        );
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(
            reopened.get().default_branch_summary_reserve_tokens,
            Some(16_384)
        );

        let mut store = SettingsStore::open(path.clone());
        store.set_default_branch_summary_reserve_tokens(None).unwrap();
        assert_eq!(store.get().default_branch_summary_reserve_tokens, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_branch_summary_reserve_tokens, None);
    }

    #[test]
    fn set_default_models_list_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_models_list, None);

        store
            .set_default_models_list(Some(vec!["claude-opus-4-8".to_string(), "gpt-5".to_string()]))
            .unwrap();
        assert_eq!(
            store.get().default_models_list,
            Some(vec!["claude-opus-4-8".to_string(), "gpt-5".to_string()])
        );
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(
            reopened.get().default_models_list,
            Some(vec!["claude-opus-4-8".to_string(), "gpt-5".to_string()])
        );

        let mut store = SettingsStore::open(path.clone());
        store.set_default_models_list(None).unwrap();
        assert_eq!(store.get().default_models_list, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_models_list, None);
    }

    #[test]
    fn set_default_skill_paths_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_skill_paths, None);

        store
            .set_default_skill_paths(Some(vec!["/opt/skills".to_string()]))
            .unwrap();
        assert_eq!(
            store.get().default_skill_paths,
            Some(vec!["/opt/skills".to_string()])
        );
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(
            reopened.get().default_skill_paths,
            Some(vec!["/opt/skills".to_string()])
        );

        let mut store = SettingsStore::open(path.clone());
        store.set_default_skill_paths(None).unwrap();
        assert_eq!(store.get().default_skill_paths, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_skill_paths, None);
    }

    #[test]
    fn set_default_prompt_template_paths_persists_and_reopening_sees_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let mut store = SettingsStore::open(path.clone());
        assert_eq!(store.get().default_prompt_template_paths, None);

        store
            .set_default_prompt_template_paths(Some(vec!["/opt/prompts".to_string()]))
            .unwrap();
        assert_eq!(
            store.get().default_prompt_template_paths,
            Some(vec!["/opt/prompts".to_string()])
        );
        let reopened = SettingsStore::open(path.clone());
        assert_eq!(
            reopened.get().default_prompt_template_paths,
            Some(vec!["/opt/prompts".to_string()])
        );

        let mut store = SettingsStore::open(path.clone());
        store.set_default_prompt_template_paths(None).unwrap();
        assert_eq!(store.get().default_prompt_template_paths, None);
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().default_prompt_template_paths, None);
    }

    // --- Feature 2 (Round 3, pi-parity): project-level `.claude/settings.json` tier.

    #[test]
    fn merge_over_prefers_the_project_value_when_both_set_a_scalar_field() {
        let global = Settings {
            default_model: Some("global-model".to_string()),
            ..Default::default()
        };
        let project = Settings {
            default_model: Some("project-model".to_string()),
            ..Default::default()
        };
        let merged = global.merge_over(&project);
        assert_eq!(merged.default_model.as_deref(), Some("project-model"));
    }

    #[test]
    fn merge_over_falls_through_to_the_global_value_when_the_project_leaves_a_field_unset() {
        let global = Settings {
            default_model: Some("global-model".to_string()),
            default_gateway_url: Some("http://global-gw".to_string()),
            ..Default::default()
        };
        let project = Settings {
            default_gateway_url: Some("http://project-gw".to_string()),
            ..Default::default()
        };
        let merged = global.merge_over(&project);
        assert_eq!(
            merged.default_model.as_deref(),
            Some("global-model"),
            "a field the project doesn't set must fall through to the global value unchanged"
        );
        assert_eq!(merged.default_gateway_url.as_deref(), Some("http://project-gw"));
    }

    #[test]
    fn merge_over_replaces_a_list_field_wholesale_rather_than_appending() {
        let global = Settings {
            default_skill_paths: Some(vec!["/global/skills".to_string()]),
            ..Default::default()
        };
        let project = Settings {
            default_skill_paths: Some(vec!["/project/skills".to_string()]),
            ..Default::default()
        };
        let merged = global.merge_over(&project);
        assert_eq!(
            merged.default_skill_paths,
            Some(vec!["/project/skills".to_string()]),
            "a project-set list must replace the global list outright, matching pi's own \
             deepMergeSettings array handling — not concatenate the two"
        );
    }

    #[test]
    fn merge_over_falls_through_to_the_global_list_when_the_project_leaves_it_unset() {
        let global = Settings {
            default_skill_paths: Some(vec!["/global/skills".to_string()]),
            ..Default::default()
        };
        let project = Settings::default();
        let merged = global.merge_over(&project);
        assert_eq!(
            merged.default_skill_paths,
            Some(vec!["/global/skills".to_string()])
        );
    }

    // pi-parity feature, Round 20: `classify_path_entry`'s four buckets — see `PathEntry`'s own doc
    // comment for the exact semantics (and how they deliberately diverge from pi's own dual-bucket bare-
    // glob behavior) each of these pins down.

    #[test]
    fn classify_path_entry_treats_a_plain_path_as_root() {
        assert!(matches!(
            classify_path_entry("/home/user/.claude/skills"),
            PathEntry::Root("/home/user/.claude/skills")
        ));
        assert!(matches!(
            classify_path_entry("relative/skills"),
            PathEntry::Root("relative/skills")
        ));
    }

    #[test]
    fn classify_path_entry_treats_a_bare_glob_as_exclude() {
        // No `!`/`+`/`-` prefix at all — a bare entry containing a glob metacharacter is still an
        // exclude filter, not a literal root to walk (see `PathEntry`'s doc comment on why this diverges
        // from pi's own "restrict to only these" bare-glob-include behavior).
        assert!(matches!(
            classify_path_entry("*.experimental.md"),
            PathEntry::Exclude("*.experimental.md")
        ));
        assert!(matches!(
            classify_path_entry("draft-?"),
            PathEntry::Exclude("draft-?")
        ));
    }

    #[test]
    fn classify_path_entry_strips_the_bang_prefix_for_exclude() {
        assert!(matches!(
            classify_path_entry("!wip-skill"),
            PathEntry::Exclude("wip-skill")
        ));
    }

    #[test]
    fn classify_path_entry_strips_the_plus_prefix_for_force_include() {
        assert!(matches!(
            classify_path_entry("+wip-skill"),
            PathEntry::Include("wip-skill")
        ));
    }

    #[test]
    fn classify_path_entry_strips_the_minus_prefix_for_force_exclude() {
        assert!(matches!(
            classify_path_entry("-wip-skill"),
            PathEntry::ForceExclude("wip-skill")
        ));
    }

    #[test]
    fn classify_path_entry_prefix_wins_even_when_the_remainder_also_has_a_glob_metacharacter() {
        // `!`/`+`/`-` always wins as the override marker, regardless of whether the remainder also
        // happens to contain `*`/`?` — matches pi's own `isOverridePattern` check running independently
        // of (and before) `hasGlobPattern`.
        assert!(matches!(
            classify_path_entry("!draft-*.md"),
            PathEntry::Exclude("draft-*.md")
        ));
    }

    #[test]
    fn effective_settings_merges_a_trusted_projects_settings_json_over_the_global_one() {
        let dir = tempfile::tempdir().unwrap();
        let project_dir = dir.path().join("project");
        fs::create_dir_all(project_dir.join(".claude")).unwrap();
        fs::write(
            project_dir.join(".claude/settings.json"),
            r#"{"default_model":"project-model"}"#,
        )
        .unwrap();

        let mut global = SettingsStore::open(dir.path().join("settings.json"));
        global
            .set_default_gateway_url(Some("http://global-gw".to_string()))
            .unwrap();

        let mut trust = crate::trust_store::TrustStore::open(dir.path().join("trusted-projects.json"));
        trust.trust(&project_dir).unwrap();

        let effective = effective_settings_with_stores(&global, &project_dir, &trust);
        assert_eq!(effective.default_model.as_deref(), Some("project-model"));
        assert_eq!(
            effective.default_gateway_url.as_deref(),
            Some("http://global-gw"),
            "a global field the project doesn't set must survive the merge"
        );
    }

    #[test]
    fn effective_settings_completely_ignores_an_untrusted_projects_settings_json() {
        let dir = tempfile::tempdir().unwrap();
        let project_dir = dir.path().join("project");
        fs::create_dir_all(project_dir.join(".claude")).unwrap();
        fs::write(
            project_dir.join(".claude/settings.json"),
            r#"{"default_model":"attacker-model","default_gateway_url":"http://evil.example"}"#,
        )
        .unwrap();

        let mut global = SettingsStore::open(dir.path().join("settings.json"));
        global
            .set_default_model(Some("global-model".to_string()))
            .unwrap();

        // Deliberately never trusted — the trust store is empty.
        let trust = crate::trust_store::TrustStore::open(dir.path().join("trusted-projects.json"));

        let effective = effective_settings_with_stores(&global, &project_dir, &trust);
        assert_eq!(
            effective.default_model.as_deref(),
            Some("global-model"),
            "an untrusted project's settings.json must be completely ignored, not partially applied"
        );
        assert_eq!(
            effective.default_gateway_url, None,
            "the untrusted project's gateway-url redirection attempt must not apply at all"
        );
    }

    #[test]
    fn effective_settings_degrades_to_global_only_when_the_project_settings_file_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let project_dir = dir.path().join("project");
        fs::create_dir_all(&project_dir).unwrap(); // no `.claude/settings.json` at all

        let mut global = SettingsStore::open(dir.path().join("settings.json"));
        global
            .set_default_model(Some("global-model".to_string()))
            .unwrap();

        let mut trust = crate::trust_store::TrustStore::open(dir.path().join("trusted-projects.json"));
        trust.trust(&project_dir).unwrap();

        let effective = effective_settings_with_stores(&global, &project_dir, &trust);
        assert_eq!(effective.default_model.as_deref(), Some("global-model"));
    }

    #[test]
    fn effective_settings_degrades_to_global_only_when_the_project_settings_file_is_malformed() {
        let dir = tempfile::tempdir().unwrap();
        let project_dir = dir.path().join("project");
        fs::create_dir_all(project_dir.join(".claude")).unwrap();
        fs::write(
            project_dir.join(".claude/settings.json"),
            "not valid json at all { [ }",
        )
        .unwrap();

        let mut global = SettingsStore::open(dir.path().join("settings.json"));
        global
            .set_default_model(Some("global-model".to_string()))
            .unwrap();

        let mut trust = crate::trust_store::TrustStore::open(dir.path().join("trusted-projects.json"));
        trust.trust(&project_dir).unwrap();

        let effective = effective_settings_with_stores(&global, &project_dir, &trust);
        assert_eq!(
            effective.default_model.as_deref(),
            Some("global-model"),
            "a malformed project settings.json must degrade to global-only, not error or panic"
        );
    }

    #[test]
    fn project_settings_path_is_dot_claude_settings_json_under_cwd() {
        let cwd = Path::new("/some/project");
        assert_eq!(
            project_settings_path(cwd),
            Path::new("/some/project/.claude/settings.json")
        );
    }
}
