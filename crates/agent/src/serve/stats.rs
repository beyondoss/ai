//! What a session reports about itself: `session_stats`, the compaction policy it shows, the live counters a
//! running prompt answers `get_state` from, and the settings block.

use crate::serve::*;

/// The compaction policy a session runs on `model`: the model's window and input limit, its
/// [`agent_core::models::ModelCaps::compact_at`], and the operator's overrides. The one
/// place `build_agent`, the pressure nudge and `context_usage` resolve it, so all three agree on where
/// compaction fires. `enabled` is left at its default; `build_agent` applies the live flag.
pub(crate) fn compaction_config(cfg: &ServeConfig, model: &str) -> agent_core::CompactionConfig {
    resolve_compaction(model, &cfg.compaction)
}

/// The operator's compaction flags, as `run` and `serve` both take them. `None`/`false` leaves the
/// model's own value or the struct default.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompactionFlags {
    /// `--context-window`.
    pub context_window: Option<u32>,
    /// `--compact-at`: `None` takes the model's cap, `Some(0)` removes it, `Some(n)` sets it.
    pub compact_at: Option<u32>,
    /// `--compaction-reserve-tokens`.
    pub reserve_tokens: Option<u32>,
    /// `--compaction-keep-recent-tokens`.
    pub keep_recent_tokens: Option<u32>,
    /// `--no-tool-masking`: summarize without masking old tool output first.
    pub no_tool_masking: bool,
    /// `--tool-masking-keep-turns`: turns whose tool output is never masked.
    pub tool_masking_keep_turns: Option<usize>,
    /// `--tool-masking-keep-tokens`: recent tokens never masked when the kept turns hold more.
    pub tool_masking_keep_tokens: Option<u32>,
    /// `--tool-masking-min-tokens`: the least a masking pass must free to be applied.
    pub tool_masking_min_tokens: Option<u32>,
    /// `--no-cached-summary`: always summarize from a re-rendered transcript.
    pub no_cached_summary: bool,
    /// `--compaction-candidates`: the changes under evaluation this run turns on.
    pub candidates: agent_core::CompactionCandidates,
}

/// [`compaction_config`] from the raw flag values, shared with `run`. `context_window` and the two
/// budgets override the model's own when given (the reserve is the struct's absolute default
/// otherwise, as before; keep-recent and the masking bar are the model family's, in real tokens —
/// see `agent_core::TokenFamily`). A `compact_at` cap also bounds `keep_recent_tokens` to half of
/// itself unless that was set explicitly: a compaction must keep less than the line that fires it, or
/// every turn re-fires.
pub fn resolve_compaction(model: &str, flags: &CompactionFlags) -> agent_core::CompactionConfig {
    let caps = agent_core::capabilities(model);
    let d = agent_core::CompactionConfig::default();
    // The model's family, price tier and family defaults.
    let m = agent_core::CompactionConfig::for_model(model);
    let keep_recent = m.family.keep_recent_tokens();
    let compact_at = flags
        .compact_at
        .map_or(caps.compact_at, |n| (n > 0).then_some(n));
    agent_core::CompactionConfig {
        context_window: flags.context_window.unwrap_or(caps.context_window),
        input_limit: caps.input_limit,
        compact_at,
        reserve_tokens: flags.reserve_tokens.unwrap_or(d.reserve_tokens),
        keep_recent_tokens: flags
            .keep_recent_tokens
            .unwrap_or_else(|| compact_at.map_or(keep_recent, |cap| keep_recent.min(cap / 2))),
        mask_tool_output: !flags.no_tool_masking,
        mask_keep_turns: flags.tool_masking_keep_turns.unwrap_or(d.mask_keep_turns),
        mask_keep_tokens: flags.tool_masking_keep_tokens,
        mask_min_free_tokens: flags
            .tool_masking_min_tokens
            .unwrap_or(m.mask_min_free_tokens),
        cached_summary: !flags.no_cached_summary,
        candidates: flags.candidates,
        family: m.family,
        price_tier: m.price_tier,
        ..d
    }
}

/// Token + step accounting for a session, including the prompt-cache reads/writes that make the
/// Anthropic cache breakpoints observable. Shared by the `prompt` response and `get_state`.
/// `model` is the session's *currently active* model, not necessarily the one that produced
/// `last_input_tokens` — a `set_model`/`cycle_model` mid-session means the context-window figure is
/// always reported against what a client would actually be constrained by on the *next* turn, matching
/// how a client would use this value (to decide whether compaction is coming soon), not as a historical
/// record of what the previous turn's provider reported. `compaction` is that model's resolved policy
/// ([`compaction_config`]). `context_window` is its input budget
/// ([`agent_core::CompactionConfig::input_budget`]): the window, or the lower input cap a GPT-5-family
/// model enforces, so `percent` reaches 100 where the vendor starts rejecting the prompt. `compact_at`
/// is where compaction actually fires ([`agent_core::CompactionConfig::trigger_tokens`]), which a
/// per-model cap puts well below that.
pub(super) fn session_stats(session: &Session, compaction: &agent_core::CompactionConfig) -> Value {
    let context_window = compaction.input_budget();
    // `last_input_tokens == 0` covers both "no turn has run yet" and pi's "immediately after
    // compaction, before a new reply" case (compaction never sets this field itself — only a real
    // provider response does) — both genuinely have no current-context figure to report, so `null`
    // rather than a misleading `0%`. Matches pi's `contextUsage` being `null` in the same situations.
    // Track L37 (pi-parity fix): `last_input_tokens` alone is only ever as fresh as the last turn's
    // own provider-reported usage — every message appended *since* then (this turn's own user prompt,
    // a mid-run tool round-trip) was missing from the reported figure entirely. `trailing_tokens` is
    // the exact same delta `should_compact`/`is_hard_overflow` already fold in before comparing against
    // the context window, so a client reading `context_usage` sees the same live-prompt size the
    // compaction trigger itself is actually watching, not a stale undercount.
    let context_usage = (session.last_input_tokens > 0).then(|| {
        let tokens =
            session
                .last_input_tokens
                .saturating_add(agent_core::compaction::trailing_tokens(
                    session,
                    compaction.family,
                ));
        json!({
            "tokens": tokens,
            "context_window": context_window,
            "compact_at": compaction.trigger_tokens(),
            "percent": (tokens as f64 / context_window as f64 * 100.0),
        })
    });
    let breakdown = message_type_breakdown(session);
    let usage_totals = message_usage_totals(session);
    json!({
        "steps": session.steps,
        "input_tokens": usage_totals.input_tokens,
        "output_tokens": usage_totals.output_tokens,
        "cache_read_tokens": usage_totals.cache_read_tokens,
        "cache_write_tokens": usage_totals.cache_write_tokens,
        "cache_write_1h_tokens": usage_totals.cache_write_1h_tokens,
        "reasoning_tokens": usage_totals.reasoning_tokens,
        "last_input_tokens": session.last_input_tokens,
        "context_usage": context_usage,
        "user_messages": breakdown.user_messages,
        "assistant_messages": breakdown.assistant_messages,
        "tool_calls": breakdown.tool_calls,
        "tool_results": breakdown.tool_results,
        "total_messages": session.messages.len(),
    })
}

/// Cumulative token usage across every message currently in `session.messages` (Task #6, pi-parity
/// fix) — computed fresh on every `session_stats` call, the same way `message_type_breakdown`, just
/// below, already derives its counts fresh from `session.messages` rather than a running counter.
///
/// Previously `session_stats` read `Session::input_tokens`/`output_tokens`/... directly — those fields
/// are a *process-lifetime* running total (`Session::record_usage` only ever adds to them) that resets
/// to zero every time the process restarts, since `SessionStore::open` only restores `session.messages`
/// from disk, never those counters. A resumed session's `get_session_stats` (and the `prompt`
/// response/`get_state`, both of which share this same function) therefore reported totals covering
/// only activity since the *current* process started, not the session's full history — silently
/// resetting on every restart with no indication anything was lost.
///
/// Same caveat `message_type_breakdown` already carries: a compacted-away message's `usage` is gone
/// along with the rest of its content, so this reports what's still visible on the active path, not a
/// truly unbounded historical total — an accepted, pre-existing limitation of deriving these figures
/// from `session.messages` at all, not something this fix introduces.
pub(super) struct MessageUsageTotals {
    pub(super) input_tokens: u64,
    pub(super) output_tokens: u64,
    pub(super) cache_read_tokens: u64,
    pub(super) cache_write_tokens: u64,
    pub(super) cache_write_1h_tokens: u64,
    pub(super) reasoning_tokens: u64,
}

pub(super) fn message_usage_totals(session: &Session) -> MessageUsageTotals {
    let mut totals = MessageUsageTotals {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        cache_write_1h_tokens: 0,
        reasoning_tokens: 0,
    };
    // A subagent's spend rides the tool-results message that carried its result (see
    // `Message::delegated_usage`): delegated work is part of what this session cost.
    for usage in session
        .messages
        .iter()
        .flat_map(|m| [m.usage, m.delegated_usage])
        .flatten()
    {
        totals.input_tokens += u64::from(usage.input_tokens);
        totals.output_tokens += u64::from(usage.output_tokens);
        totals.cache_read_tokens += u64::from(usage.cache_read_tokens);
        totals.cache_write_tokens += u64::from(usage.cache_write_tokens);
        totals.cache_write_1h_tokens += u64::from(usage.cache_write_1h_tokens);
        totals.reasoning_tokens += u64::from(usage.reasoning_tokens);
    }
    totals
}

/// [`message_usage_totals`], reshaped into [`crate::export::UsageTotals`] — Fix 6 (pi-parity gap): the
/// standalone `export` subcommand (`main.rs`) has no live `Agent`/`Session` with running counters to
/// pull from (a freshly `SessionStore::open`ed session never restores those — see
/// `message_usage_totals`'s own doc comment for why), but it does have `sess.messages` right there,
/// from which this is fully reconstructable — no live agent needed. Previously that entry point passed
/// `usage: None` unconditionally, the one of the three export entry points (alongside `serve`'s
/// `export_html` RPC and `run --export`, both of which already report real totals from a live
/// `Session`'s running counters) that silently omitted the stats section's usage line entirely.
pub fn message_export_usage_totals(session: &Session) -> crate::export::UsageTotals {
    let totals = message_usage_totals(session);
    crate::export::UsageTotals {
        input_tokens: totals.input_tokens,
        output_tokens: totals.output_tokens,
        cache_read_tokens: totals.cache_read_tokens,
        cache_write_tokens: totals.cache_write_tokens,
    }
}

/// pi's `userMessages`/`assistantMessages`/`toolCalls`/`toolResults` (`getSessionStats`) — a client
/// wanting these previously had to fetch `get_messages` and count client-side.
///
/// Adapted, not ported 1:1: pi's message model has a dedicated `toolResult` role, separate from `user`,
/// so its `userMessages`/`toolResults` split cleanly on role alone. Ours carries a tool result as a
/// `ContentBlock::ToolResult` *within* a `User`-role message (Anthropic's own convention) — and a
/// steered mid-run text injection can land in that very same message alongside the tool results (see
/// `steering.rs`'s module doc: "folded onto the same tool-results user turn"). So a `User` message here
/// counts as a "user message" when it carries anything *other than* a tool result (real authored
/// content), and `tool_calls`/`tool_results` count individual content blocks rather than whole messages
/// — matching pi's own `toolCalls` definition, which already counts blocks within an assistant message,
/// not messages themselves.
pub(super) struct MessageTypeBreakdown {
    pub(super) user_messages: u64,
    pub(super) assistant_messages: u64,
    pub(super) tool_calls: u64,
    pub(super) tool_results: u64,
}

pub(super) fn message_type_breakdown(session: &Session) -> MessageTypeBreakdown {
    let mut breakdown = MessageTypeBreakdown {
        user_messages: 0,
        assistant_messages: 0,
        tool_calls: 0,
        tool_results: 0,
    };
    for m in session.messages.iter() {
        match m.role {
            agent_core::Role::User => {
                let has_non_tool_result = m
                    .content
                    .iter()
                    .any(|c| !matches!(c, agent_core::ContentBlock::ToolResult { .. }));
                if has_non_tool_result {
                    breakdown.user_messages += 1;
                }
                breakdown.tool_results += m
                    .content
                    .iter()
                    .filter(|c| matches!(c, agent_core::ContentBlock::ToolResult { .. }))
                    .count() as u64;
            }
            agent_core::Role::Assistant => {
                breakdown.assistant_messages += 1;
                breakdown.tool_calls += m
                    .content
                    .iter()
                    .filter(|c| matches!(c, agent_core::ContentBlock::ToolUse { .. }))
                    .count() as u64;
            }
            agent_core::Role::System => {}
        }
    }
    breakdown
}

/// Insert `session_id`/`session_file` into `m` — the on-disk identity every `get_state`/
/// `get_session_stats` response carries, idle or busy (Fix 8, pi-parity gap: `get_session_stats`
/// previously omitted both in its idle response, and its entire busy response, unlike `get_state`'s own
/// sibling arms, which already backfilled them from these same two sources). Shared so the two RPC
/// types can't drift out of shape with each other again.
/// In service mode `session_file` reports `null`: the path is a mount on
/// the replica, which is this tenant's storage but not its filesystem — nothing on the other end can
/// open it, and publishing it would leak the fleet's layout. The id is the address that matters.
pub(super) fn insert_session_identity(
    m: &mut Map<String, Value>,
    persistence: &Persistence,
    mode: &Mode,
) {
    m.insert("session_id".into(), json!(persistence.session_id()));
    m.insert(
        "session_file".into(),
        if mode.reports_session_file() {
            json!(persistence.session_file().map(|p| p.display().to_string()))
        } else {
            Value::Null
        },
    );
}

/// The runtime-mutable settings and queue depth `get_state` reports — pi's `get_state` carries the
/// same shape (`thinkingLevel`/`autoCompactionEnabled`/`steeringMode`/`followUpMode`/
/// `pendingMessageCount`), so a client can render current settings and "N queued" without a second
/// round trip. Answerable from the process's own mutable state, not `&Session`, so — like
/// `session_stats` — it's available even mid-run (see the `prompt` arm's read-only-command handling).
///
/// Fix 5 (pi-parity gap): `steer_queue`/`follow_up_queue` carry the queued lanes' actual message text
/// (`Steering::steer_texts`/`follow_up_texts`), not just `pending_messages`' bare count — previously a
/// client had no way to see *what* was queued short of guessing from its own prior `steer`/`follow_up`
/// calls, unlike pi's own `queue_update` event (`steering`/`followUp` string arrays). No unsolicited
/// push-time event is added alongside it (that gap is deliberate — see `steering.rs`'s own module doc
/// comment); instead every place this function's result already lands (`get_state`, and the `steer`/
/// `follow_up` handlers' own ack responses, below) picks the content up for free.
pub(super) fn runtime_settings(
    current_level: agent_core::ThinkingLevel,
    current_auto_compaction: bool,
    current_auto_retry: bool,
    current_block_images: bool,
    current_image_auto_resize: bool,
    steering: &agent_core::Steering,
) -> Value {
    json!({
        "thinking_level": current_level.as_str(),
        "auto_compaction": current_auto_compaction,
        "auto_retry": current_auto_retry,
        // Pass 20 (pi-parity fix): reported here for the same reason `auto_compaction`/`auto_retry`
        // are — both are now live-toggleable (`set_block_images`/`set_image_auto_resize`), so `get_state`
        // needs to reflect whichever value is actually in effect, not just what `serve` started with.
        "block_images": current_block_images,
        "image_auto_resize": current_image_auto_resize,
        "steering_mode": match steering.steering_mode() {
            agent_core::QueueMode::OneAtATime => "one_at_a_time",
            agent_core::QueueMode::All => "all",
        },
        "follow_up_mode": match steering.follow_up_mode() {
            agent_core::QueueMode::OneAtATime => "one_at_a_time",
            agent_core::QueueMode::All => "all",
        },
        "pending_messages": steering.pending_count(),
        "steer_queue": steering.steer_texts(),
        "follow_up_queue": steering.follow_up_texts(),
    })
}

/// `{steer_queue, follow_up_queue}` alone — the same pair [`runtime_settings`] carries, factored out
/// so the `steer`/`follow_up` handlers' own ack responses (Fix 5, pi-parity gap) can attach it without
/// pulling in that function's wider signature (`thinking_level`/`auto_compaction`/… — irrelevant to a
/// bare queue push/ack).
pub(super) fn queue_content(steering: &agent_core::Steering) -> Value {
    json!({
        "steer_queue": steering.steer_texts(),
        "follow_up_queue": steering.follow_up_texts(),
    })
}

/// Recover the exit code the host `bash` RPC command's terminal response reports (Fix 7, pi-parity
/// gap) — `tools::bash` never returns one structurally; a non-zero exit is folded into `result_text`
/// itself as a trailing `"Command exited with code <N>"` status line (`tools::bash::append_status`),
/// the only place it's recorded at all. `Some(0)` for an ordinary success: `tools::bash` itself treats
/// exit `0` and "no code reported" (`result.code: None`) as the identical `Ok` outcome, so neither is
/// distinguishable from the other here either. `None` for a cancelled/timed-out run — pi's own
/// `BashResult.exitCode` is `undefined` for both (see `bash_result_was_cancelled`), matching the
/// dedicated "cancelled" status text `tools::bash` appends instead of an exit-code one for either case.
pub(super) fn bash_exit_code_from_status_line(result_text: &str, is_error: bool) -> Option<i64> {
    if !is_error {
        return Some(0);
    }
    let marker = "Command exited with code ";
    let start = result_text.rfind(marker)? + marker.len();
    result_text[start..]
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .filter(|digits| !digits.is_empty())
        .and_then(|digits| digits.parse().ok())
}

/// Whether the host `bash` RPC command's result represents a cancelled or timed-out run rather than an
/// ordinary completion (Fix 7, pi-parity gap) — pi's own `BashResult.cancelled` conflates both under one
/// signal (`executeBashWithOperations`'s `options?.signal?.aborted` fires for either). Recovered from
/// the status line: `tools::bash`'s `"Command cancelled"`, the `bash` handler's own `"Command timed
/// out after <N> seconds"` (the client's `timeout_ms`, enforced as a cancel) or the runner's (an exec
/// endpoint's deadline), or this RPC handler's own fallback synthetic `"cancelled"` error for a tool
/// that isn't internally cancellation-aware (see the `bash` handler's own `select!` comment on why, in
/// practice, `tools::bash` always wins that race first).
pub(super) fn bash_result_was_cancelled(result_text: &str, is_error: bool) -> bool {
    is_error
        && (result_text.contains("Command cancelled")
            || result_text.contains("Command timed out after")
            || result_text == "cancelled")
}

/// A live mirror of [`Session`]'s token/step counters, updated from the event sink as a `prompt` runs —
/// so `get_state`/`get_session_stats` can answer with real progress from *inside* the busy-loop
/// (see the `prompt` arm), without the `&mut Session` the run itself holds exclusively for the turn's
/// duration. Seeded from the session's own values right before the run starts, then kept current by
/// mirroring exactly the events a client watching the stream already sees: `record_usage` matches
/// `Session::record_usage`'s accumulation, and `set_steps` takes `AgentEvent::TurnEnd`'s `step` field,
/// which is already `session.steps` post-increment at the moment that event fires. Lock-free (plain
/// atomics): the sink runs synchronously inside the run's own task, and a command handler reads it from
/// the control loop's task — never contended enough to need more than `Relaxed` ordering, since these
/// are independent counters with no cross-field invariant a reader depends on.
#[derive(Default)]
pub(super) struct LiveStats {
    pub(super) steps: AtomicU32,
    pub(super) input_tokens: std::sync::atomic::AtomicU64,
    pub(super) output_tokens: std::sync::atomic::AtomicU64,
    pub(super) cache_read_tokens: std::sync::atomic::AtomicU64,
    pub(super) cache_write_tokens: std::sync::atomic::AtomicU64,
    pub(super) cache_write_1h_tokens: std::sync::atomic::AtomicU64,
    pub(super) reasoning_tokens: std::sync::atomic::AtomicU64,
    pub(super) last_input_tokens: AtomicU32,
    /// B-L1 pi-parity gap (fixed): pi's `agent.state.pendingToolCalls` (a live, in-process reactive
    /// set) has no RPC equivalent — a client watching a long tool-heavy turn had to reconstruct "which
    /// calls are still running" itself from the raw `tool_start`/`tool_end` event stream. Mirrored from
    /// those same events (see the `prompt` command's busy-loop sink), so `get_state`/
    /// `get_session_stats` can answer it directly mid-run, the same way every other field here already
    /// does. A `Mutex<BTreeSet>` rather than another atomic: ids are strings, and insert/remove need to
    /// be atomic *with each other*, not just individually.
    pub(super) pending_tool_ids: std::sync::Mutex<std::collections::BTreeSet<String>>,
    /// The model's current `todo` list, mirrored off the `todo` tool's own `ToolProgress` events for
    /// exactly the same reason as `pending_tool_ids` above: `&session` is exclusively borrowed by the
    /// in-flight turn, so `get_todos` could otherwise only be answered while idle — and the client that
    /// most needs it is a phone attaching *mid-run*, which has already missed every `tool_progress`
    /// frame streamed before it connected. Seeded from the session (see [`current_todos`]) so a
    /// `get_todos` answered one event into a fresh turn still reports the plan the last turn left.
    pub(super) todos: std::sync::Mutex<Option<Value>>,
}

impl LiveStats {
    /// Seed from a session's current cumulative totals, so a `get_state`/`get_session_stats` answered
    /// one event into a brand-new turn still reflects everything before it, not a reset-to-zero count.
    ///
    /// Task #6 (pi-parity fix): seeded from [`message_usage_totals`] — the same per-message sum
    /// `session_stats` itself now uses — rather than `Session::input_tokens`/`output_tokens`/... those
    /// fields are only ever a *process-lifetime* running total (`Session::record_usage`'s own doc
    /// comment), reset to zero on every restart since `SessionStore::open` never restores them from
    /// disk. Seeding from the stale counters here meant the mid-run "busy" snapshot a client polls via
    /// `get_state`/`get_session_stats` while a `prompt` is in flight (see the `prompt` arm's own
    /// busy-loop) reported only the *current process's* activity on a resumed session, same bug, same
    /// fix, different call site.
    pub(super) fn from_session(session: &Session) -> Self {
        let totals = message_usage_totals(session);
        Self {
            steps: AtomicU32::new(session.steps),
            input_tokens: totals.input_tokens.into(),
            output_tokens: totals.output_tokens.into(),
            cache_read_tokens: totals.cache_read_tokens.into(),
            cache_write_tokens: totals.cache_write_tokens.into(),
            cache_write_1h_tokens: totals.cache_write_1h_tokens.into(),
            reasoning_tokens: totals.reasoning_tokens.into(),
            last_input_tokens: AtomicU32::new(session.last_input_tokens),
            pending_tool_ids: std::sync::Mutex::new(std::collections::BTreeSet::new()),
            todos: std::sync::Mutex::new(current_todos(session)),
        }
    }

    /// Record the list a `todo` call just committed (the tool's own `ToolProgress` payload).
    pub(super) fn todos_updated(&self, todos: Value) {
        *self.todos.lock().unwrap_or_else(|e| e.into_inner()) = Some(todos);
    }

    pub(super) fn todos(&self) -> Option<Value> {
        self.todos.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub(super) fn tool_started(&self, id: String) {
        self.pending_tool_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id);
    }

    pub(super) fn tool_ended(&self, id: &str) {
        self.pending_tool_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id);
    }

    /// Fold one turn's usage into the running totals — the same arithmetic as
    /// [`Session::record_usage`], applied to atomics instead of plain fields.
    pub(super) fn record_usage(&self, usage: &agent_core::TokenUsage) {
        self.record_delegated_usage(usage);
        let last = usage
            .input_tokens
            .saturating_add(usage.cache_read_tokens)
            .saturating_add(usage.cache_write_tokens);
        self.last_input_tokens.store(last, Ordering::Relaxed);
    }

    /// Fold a subagent's request into the running totals only — [`Session::record_delegated_usage`]'s
    /// arithmetic. Never the live context size: the child's prompt is not this session's.
    pub(super) fn record_delegated_usage(&self, usage: &agent_core::TokenUsage) {
        self.input_tokens
            .fetch_add(usage.input_tokens.into(), Ordering::Relaxed);
        self.output_tokens
            .fetch_add(usage.output_tokens.into(), Ordering::Relaxed);
        self.cache_read_tokens
            .fetch_add(usage.cache_read_tokens.into(), Ordering::Relaxed);
        self.cache_write_tokens
            .fetch_add(usage.cache_write_tokens.into(), Ordering::Relaxed);
        self.cache_write_1h_tokens
            .fetch_add(usage.cache_write_1h_tokens.into(), Ordering::Relaxed);
        self.reasoning_tokens
            .fetch_add(usage.reasoning_tokens.into(), Ordering::Relaxed);
    }

    pub(super) fn set_steps(&self, step: u32) {
        self.steps.store(step, Ordering::Relaxed);
    }

    /// A `session_stats`-shaped snapshot of the current values.
    pub(super) fn snapshot(&self) -> Value {
        json!({
            "steps": self.steps.load(Ordering::Relaxed),
            "input_tokens": self.input_tokens.load(Ordering::Relaxed),
            "output_tokens": self.output_tokens.load(Ordering::Relaxed),
            "cache_read_tokens": self.cache_read_tokens.load(Ordering::Relaxed),
            "cache_write_tokens": self.cache_write_tokens.load(Ordering::Relaxed),
            "cache_write_1h_tokens": self.cache_write_1h_tokens.load(Ordering::Relaxed),
            "reasoning_tokens": self.reasoning_tokens.load(Ordering::Relaxed),
            "last_input_tokens": self.last_input_tokens.load(Ordering::Relaxed),
            "pending_tool_ids": self
                .pending_tool_ids
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `model`'s compaction policy with no operator overrides.
    fn test_compaction(model: &str) -> agent_core::CompactionConfig {
        resolve_compaction(model, &CompactionFlags::default())
    }

    #[test]
    fn context_usage_reports_where_compaction_fires_with_every_override() {
        let mut session = Session::new();
        session.last_input_tokens = 10_000;
        let at =
            |c: agent_core::CompactionConfig| session_stats(&session, &c)["context_usage"].clone();
        // A capped model: its cap, under a 1M input budget.
        let u = at(test_compaction("claude-opus-5-5"));
        assert_eq!(u["context_window"], 1_000_000);
        assert_eq!(u["compact_at"], 200_000);
        // An unlisted model: the 200K fallback.
        assert_eq!(
            at(test_compaction("claude-opus-4-8"))["compact_at"],
            200_000
        );
        // `--compact-at n` sets it, `--compact-at 0` removes the model's cap.
        let pinned = resolve_compaction(
            "claude-opus-5-5",
            &CompactionFlags {
                compact_at: Some(150_000),
                ..Default::default()
            },
        );
        assert_eq!(at(pinned)["compact_at"], 150_000);
        let off = resolve_compaction(
            "claude-opus-5-5",
            &CompactionFlags {
                compact_at: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(at(off)["compact_at"], 1_000_000 - 16_384);
        // `--context-window` still bounds both figures; the lower of the two lines wins.
        let small = resolve_compaction(
            "claude-opus-5-5",
            &CompactionFlags {
                context_window: Some(100_000),
                ..Default::default()
            },
        );
        assert_eq!(at(small)["context_window"], 100_000);
        assert_eq!(at(small)["compact_at"], 100_000 - 16_384);
    }

    #[test]
    fn a_small_compact_at_keeps_less_than_it_fires_at() {
        // keep_recent must stay under the trigger, or every compaction lands back over it.
        let c = resolve_compaction(
            "claude-opus-5-5",
            &CompactionFlags {
                compact_at: Some(30_000),
                ..Default::default()
            },
        );
        assert_eq!(c.keep_recent_tokens, 15_000);
        // The model defaults are all above 2 × keep_recent: the family's own value, in real tokens
        // (Haiku's 90K cap leaves room for Claude's 40.5K; GPT keeps 22K).
        assert_eq!(
            test_compaction("claude-haiku-5-5").keep_recent_tokens,
            40_500
        );
        assert_eq!(test_compaction("gpt-6-luna").keep_recent_tokens, 22_000);
        assert_eq!(test_compaction("kimi-k3").keep_recent_tokens, 20_000);
        // The family and its price tier ride along.
        let haiku = test_compaction("claude-haiku-5-5");
        assert_eq!(haiku.family, agent_core::TokenFamily::Claude);
        assert_eq!(haiku.price_tier, Some(100_000));
        assert_eq!(haiku.mask_min_free(), 14_400);
        // An explicit keep-recent is the operator's call.
        let c = resolve_compaction(
            "claude-opus-5-5",
            &CompactionFlags {
                compact_at: Some(30_000),
                keep_recent_tokens: Some(25_000),
                ..Default::default()
            },
        );
        assert_eq!(c.keep_recent_tokens, 25_000);
    }

    #[test]
    fn session_stats_reports_no_context_usage_before_any_turn_has_run() {
        // pi: agent-session-stats.test.ts — `contextUsage` is `null` when nothing has run yet (and,
        // by the same field, immediately after a compaction that hasn't been followed by a real reply
        // — compaction never sets `last_input_tokens` itself, only a real provider response does).
        let session = Session::new();
        let stats = session_stats(&session, &test_compaction("claude-opus-4-8"));
        assert_eq!(stats["context_usage"], Value::Null, "got: {stats}");
    }

    #[test]
    fn session_stats_reports_the_message_type_breakdown() {
        // pi-parity fix (M14): `get_session_stats` had no `userMessages`/`assistantMessages`/
        // `toolCalls`/`toolResults`/`totalMessages` at all — a client wanting these had to fetch
        // `get_messages` and count client-side.
        let mut session = Session::new();
        session.user("first request");
        session.push(agent_core::Message::assistant(vec![
            agent_core::ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "a.rs" }),
                thought_signature: None,
            },
            agent_core::ContentBlock::ToolUse {
                id: "2".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "b.rs" }),
                thought_signature: None,
            },
        ]));
        // A pure tool-results turn — no user-authored content of its own.
        session.push(agent_core::Message {
            role: agent_core::Role::User,
            content: vec![
                agent_core::ContentBlock::ToolResult {
                    tool_use_id: "1".into(),
                    content: "contents of a.rs".into(),
                    is_error: false,
                    images: vec![],
                },
                agent_core::ContentBlock::ToolResult {
                    tool_use_id: "2".into(),
                    content: "contents of b.rs".into(),
                    is_error: false,
                    images: vec![],
                },
            ],
            model_id: None,
            error_message: None,
            aborted: false,
            usage: None,
            stop_reason: None,
            prior_turn_tokens: None,
            delegated_usage: None,
        });
        session.push(agent_core::Message::assistant(vec![
            agent_core::ContentBlock::text("done reading both files"),
        ]));
        session.user("second request");

        let stats = session_stats(&session, &test_compaction("claude-opus-4-8"));
        assert_eq!(stats["total_messages"], 5, "got: {stats}");
        assert_eq!(
            stats["user_messages"], 2,
            "the pure tool-results turn must not count as a user message: {stats}"
        );
        assert_eq!(stats["assistant_messages"], 2, "got: {stats}");
        assert_eq!(stats["tool_calls"], 2, "got: {stats}");
        assert_eq!(stats["tool_results"], 2, "got: {stats}");
    }

    #[test]
    fn session_stats_a_steered_tool_results_turn_still_counts_as_a_user_message() {
        // The one case a `User` message mixes a tool result with genuine authored content: a steer
        // injected mid-run lands on the same turn as that turn's own tool results (see `steering.rs`'s
        // module doc). That message must count as a user message — it does carry real content.
        let mut session = Session::new();
        session.push(agent_core::Message {
            role: agent_core::Role::User,
            content: vec![
                agent_core::ContentBlock::ToolResult {
                    tool_use_id: "1".into(),
                    content: "result".into(),
                    is_error: false,
                    images: vec![],
                },
                agent_core::ContentBlock::text("actually, also check the tests"),
            ],
            model_id: None,
            error_message: None,
            aborted: false,
            usage: None,
            stop_reason: None,
            prior_turn_tokens: None,
            delegated_usage: None,
        });

        let stats = session_stats(&session, &test_compaction("claude-opus-4-8"));
        assert_eq!(stats["user_messages"], 1, "got: {stats}");
        assert_eq!(stats["tool_results"], 1, "got: {stats}");
    }

    #[test]
    fn session_stats_reports_context_usage_against_the_currently_active_model() {
        let mut session = Session::new();
        session.last_input_tokens = 50_000;
        let stats = session_stats(&session, &test_compaction("claude-opus-4-8"));
        let usage = &stats["context_usage"];
        assert_eq!(usage["tokens"], 50_000);
        let context_window = agent_core::models::capabilities("claude-opus-4-8").context_window;
        assert_eq!(usage["context_window"], context_window);
        let expected_percent = 50_000.0 / context_window as f64 * 100.0;
        assert!(
            (usage["percent"].as_f64().unwrap() - expected_percent).abs() < f64::EPSILON,
            "got: {usage}"
        );
    }

    #[test]
    fn session_stats_context_usage_folds_in_trailing_tokens_since_the_last_usage_snapshot() {
        // Track L37 (pi-parity fix): `context_usage` used to report `last_input_tokens` alone, which
        // is only ever as fresh as the last turn's own provider-reported usage — every message
        // appended *since* (this turn's own prompt, a mid-run tool round-trip) was invisible to a
        // client reading this field, even though `should_compact`/`is_hard_overflow` already fold the
        // same `agent_core::compaction::trailing_tokens` delta in before comparing against the window.
        let mut session = Session::new();
        session.last_input_tokens = 50_000;
        session.last_usage_message_count = session.messages.len(); // snapshot taken with no messages yet
        let stats_before = session_stats(&session, &test_compaction("claude-opus-4-8"));
        assert_eq!(
            stats_before["context_usage"]["tokens"], 50_000,
            "no trailing messages yet: {stats_before}"
        );

        // ~100 estimated tokens' worth of message appended since the snapshot (400 chars / 4).
        session.push(agent_core::Message::user("x".repeat(400)));
        let stats_after = session_stats(&session, &test_compaction("claude-opus-4-8"));
        let tokens_after = stats_after["context_usage"]["tokens"].as_u64().unwrap();
        assert!(
            tokens_after > 50_000,
            "trailing tokens since the last usage snapshot must be folded into `context_usage`: {stats_after}"
        );
        let context_window = agent_core::models::capabilities("claude-opus-4-8").context_window;
        let expected_percent = tokens_after as f64 / context_window as f64 * 100.0;
        assert!(
            (stats_after["context_usage"]["percent"].as_f64().unwrap() - expected_percent).abs()
                < f64::EPSILON,
            "percent must be derived from the same trailing-inclusive figure, not just last_input_tokens: {stats_after}"
        );
    }

    #[test]
    fn session_stats_context_window_reflects_a_model_switch_not_the_original_model() {
        // The window reported is always for the model the *next* turn would actually run against —
        // a session that ran under one model and then switched (`set_model`/`cycle_model`) must report
        // the new model's window, not the one that produced `last_input_tokens`.
        let mut session = Session::new();
        session.last_input_tokens = 10_000;
        let stats = session_stats(&session, &test_compaction("gpt-5-mini"));
        // gpt-5-mini's 400K window caps input at 272K: the figure compaction compares against.
        assert_eq!(stats["context_usage"]["context_window"], 272_000);
    }

    #[test]
    fn session_stats_sums_usage_from_message_history_not_the_process_lifetime_counter() {
        // Task #6 (pi-parity fix): a resumed session's `Session::input_tokens`/`output_tokens`/...
        // reset to zero every process restart (`SessionStore::open` only restores `session.messages`,
        // never those running counters) — simulated here by setting them directly while leaving
        // `session.messages` carrying the real historical `usage` a persisted session would actually
        // have. `session_stats` must report the totals from the message history, not the
        // process-lifetime counters (which this test deliberately sets to an obviously-wrong sentinel
        // to prove they're ignored).
        let mut session = Session::new();
        session.user("go");
        session.push(
            agent_core::Message::assistant(vec![agent_core::ContentBlock::text("ok")])
                .with_model_id("claude-opus-4-8")
                .with_usage(agent_core::TokenUsage {
                    input_tokens: 100,
                    output_tokens: 20,
                    cache_read_tokens: 5,
                    cache_write_tokens: 3,
                    cache_write_1h_tokens: 1,
                    reasoning_tokens: 2,
                }),
        );
        session.push(
            agent_core::Message::assistant(vec![agent_core::ContentBlock::text("more")])
                .with_model_id("claude-opus-4-8")
                .with_usage(agent_core::TokenUsage {
                    input_tokens: 50,
                    output_tokens: 10,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    cache_write_1h_tokens: 0,
                    reasoning_tokens: 0,
                }),
        );
        // A "reset on restart" process-lifetime counter simulated as a wrong, stale value that must
        // not leak into the reported totals.
        session.input_tokens = 999_999;
        session.output_tokens = 999_999;

        let stats = session_stats(&session, &test_compaction("claude-opus-4-8"));
        assert_eq!(stats["input_tokens"], 150, "got: {stats}");
        assert_eq!(stats["output_tokens"], 30, "got: {stats}");
        assert_eq!(stats["cache_read_tokens"], 5, "got: {stats}");
        assert_eq!(stats["cache_write_tokens"], 3, "got: {stats}");
        assert_eq!(stats["cache_write_1h_tokens"], 1, "got: {stats}");
        assert_eq!(stats["reasoning_tokens"], 2, "got: {stats}");
    }

    #[test]
    fn session_stats_reports_full_history_after_a_simulated_restart() {
        // The exact scenario Task #6 names: persist a session with several messages carrying usage,
        // reload it fresh (simulating a process restart via a real `SessionStore`/`Persistence` round
        // trip, not just constructing a `Session` by hand), then confirm `session_stats` reflects the
        // full history rather than zero/current-process-only.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, mut session) =
            Persistence::open_repo(dir.path(), "/w", "claude-opus-4-8", SessionSelect::Latest)
                .unwrap();
        session.user("go");
        session.push(
            agent_core::Message::assistant(vec![agent_core::ContentBlock::text("ok")])
                .with_model_id("claude-opus-4-8")
                .with_usage(agent_core::TokenUsage {
                    input_tokens: 1_000,
                    output_tokens: 200,
                    ..Default::default()
                }),
        );
        persistence.persist(&session).unwrap();
        drop(persistence);
        drop(session);

        // "Restart": a fresh `Persistence::open_repo` — a brand-new `Session::new()` with none of the
        // in-memory counters the original process accumulated (matching a real process restart exactly,
        // since those counters never persist regardless).
        let (_restarted, reloaded) =
            Persistence::open_repo(dir.path(), "/w", "claude-opus-4-8", SessionSelect::Latest)
                .unwrap();
        assert_eq!(
            reloaded.input_tokens, 0,
            "sanity: the running counter itself stayed at zero"
        );

        let stats = session_stats(&reloaded, &test_compaction("claude-opus-4-8"));
        assert_eq!(
            stats["input_tokens"], 1_000,
            "must reflect the full persisted history, not the fresh process's zeroed counter: {stats}"
        );
        assert_eq!(stats["output_tokens"], 200, "got: {stats}");
    }

    #[test]
    fn live_stats_from_session_seeds_from_message_history_not_the_process_lifetime_counter() {
        // Same Task #6 fix, for the mid-run "busy" snapshot path (`LiveStats::from_session`, consulted
        // by `get_state`/`get_session_stats` while a `prompt` is in flight) — it must seed from the same
        // per-message totals `session_stats` uses, not the process-lifetime `Session` counters.
        let mut session = Session::new();
        session.push(
            agent_core::Message::assistant(vec![agent_core::ContentBlock::text("ok")])
                .with_model_id("claude-opus-4-8")
                .with_usage(agent_core::TokenUsage {
                    input_tokens: 42,
                    output_tokens: 7,
                    ..Default::default()
                }),
        );
        session.input_tokens = 999_999; // stale process-lifetime counter, must be ignored

        let live = LiveStats::from_session(&session);
        let snapshot = live.snapshot();
        assert_eq!(snapshot["input_tokens"], 42, "got: {snapshot}");
        assert_eq!(snapshot["output_tokens"], 7, "got: {snapshot}");
    }

    // Fix 1 (pi-parity bug): `default_project_trust` used to be checked *before* an explicit per-path
    // `TrustStore` entry, so an operator's specific `agent trust`/`agent untrust <path>` exception could
    // be silently overridden by a coarser blanket policy. `resolve_project_trust` is the shared
    // precedence both `run` and `serve` now consult, taking the already-resolved `Trust` lookup as a
    // plain parameter rather than reading the real trust store, so these tests don't need to sandbox any
    // filesystem/global state.
}
