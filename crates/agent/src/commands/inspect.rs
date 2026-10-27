//! Read-only listings and renders: `tools`, `list-models`, `export`.

use std::path::PathBuf;

use beyond_agent::{serve, tools};

use crate::{format_token_count, fuzzy_match};

/// `tools`: the advertised tool set, as JSON definitions.
pub(crate) fn tools() -> Result<(), Box<dyn std::error::Error>> {
    let mut reg = tools::default_registry();
    tools::apply_filter(&mut reg, None, None, false);
    println!("{} tools:\n", reg.len());
    println!("{}", serde_json::to_string_pretty(&reg.definitions())?);
    Ok(())
}

/// `list-models [search]`: the capability table, fuzzy-filtered by `search`.
pub(crate) fn list_models(search: Option<String>) {
    // Pi-parity fix: previously a bare list of ids — pi's own `--list-models` prints a table
    // (provider/model/context/max-out/thinking/images) built from data its model catalogue
    // already carries. Beyond has no separate provider field (a model id is forwarded verbatim;
    // see `agent_core::models`'s own module doc comment), so this mirrors the rest of pi's
    // columns from `agent_core::capabilities`, which already computes every one of them for
    // wire-shaping — nothing new is invented here, just surfaced.
    //
    // Task #51 (pi-parity fix): an optional positional `search` fuzzy-filters model ids —
    // matches pi's own `--list-models <search>` (`fuzzyFilter`/`fuzzyMatch`,
    // `packages/tui/src/fuzzy.ts`), a non-contiguous, order-preserving, word-boundary-scored
    // subsequence match rather than a plain substring check, so e.g. "sn5" finds
    // "claude-sonnet-4-5" the way pi's own table search does. Previously a plain
    // case-insensitive `contains`, which that query would never match at all.
    let models: Vec<&str> = match &search {
        Some(query) => {
            let mut scored: Vec<(&str, f64)> = serve::listed_models()
                .iter()
                .filter_map(|m| fuzzy_match(query, m).map(|score| (*m, score)))
                .collect();
            // Lower score is a better match (mirrors pi's own ascending sort) — a stable sort
            // keeps `listed_models()`'s own relative order as the tie-break, same as pi's
            // `Array.prototype.sort` (stable per spec).
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.into_iter().map(|(m, _)| m).collect()
        }
        None => serve::listed_models().to_vec(),
    };
    println!(
        "{:<22} {:>10} {:>10} {:>9} {:<8} {:<6}",
        "model", "context", "compact-at", "max-out", "thinking", "vision"
    );
    for model in models {
        let caps = agent_core::capabilities(model);
        let thinking =
            caps.reasoning_effort || caps.thinking != agent_core::models::ThinkingShape::None;
        // Where compaction fires with no overrides: the model's cap, else window − reserve.
        let trigger = agent_core::CompactionConfig::for_model(model).trigger_tokens();
        println!(
            "{:<22} {:>10} {:>10} {:>9} {:<8} {:<6}",
            model,
            format_token_count(caps.context_window),
            format_token_count(trigger),
            format_token_count(caps.max_output),
            if thinking { "yes" } else { "no" },
            if caps.supports_vision { "yes" } else { "no" },
        );
    }
}

/// `export <session.jsonl> [output.html]`: render a persisted session straight off disk.
pub(crate) fn export(
    session: String,
    output: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (store, sess) = beyond_agent::session_store::SessionStore::open(PathBuf::from(&session))
        .map_err(|e| format!("failed to open session {session}: {e}"))?;
    let branches = store.abandoned_branches();
    // `export_html_full` (Task #44 integration), but with `system_prompt`/`tools` genuinely
    // `None`: this standalone subcommand renders an already-persisted session file straight off
    // disk with no gateway/key/model involved at all (see this crate's own ARCHITECTURE.md), so
    // there's no live `Agent`/`ToolRegistry` here to pull either from — and the session file
    // itself records neither the exact system prompt text nor which `--tools`/`--exclude-tools`
    // filter (if any) a past run used, so reconstructing either would mean fabricating data
    // that may not match what actually ran. Usage totals are different: Fix 6 (pi-parity gap):
    // `sess`'s own running token counters (`input_tokens`/etc.) are never persisted/restored
    // across a process restart, but they're fully reconstructable by summing each message's own
    // `usage` (`serve::message_export_usage_totals` — the same computation `serve`'s
    // `export_html` RPC and `run --export` derive their own totals from, one layer up), which
    // *is* right here in `sess.messages` — previously this passed `usage: None` unconditionally,
    // the one of the three export entry points that silently omitted the line instead.
    let path = beyond_agent::export::export_html_full(
        store.meta(),
        &sess.messages,
        &branches,
        Some(beyond_agent::serve::message_export_usage_totals(&sess)),
        store.export_events(),
        None,
        None,
        output.as_deref(),
    )?;
    println!("Exported to: {}", path.display());
    Ok(())
}
