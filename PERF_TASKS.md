# Perf remediation — task plan

Companion to `PERF_AUDIT.md`. Two buckets. IDs `[T<track>-F<n>]` reference the audit.

## Measurement protocol (for MEASURED tasks)
1. Ensure a `divan` bench exercises the exact path (extend an existing bench, or add one). Benches use divan's `AllocProfiler` → reports **allocs/op** alongside **ns/op**.
2. Capture **baseline on the clean tree** (before any edit for that path): `cargo bench -p <crate> --bench <name> -- <filter>` → save to `bench/baseline-<task>.txt`.
3. Apply the change.
4. Re-run the same bench → `bench/after-<task>.txt`.
5. **Accept** only if allocs/op and/or ns/op drop materially **and** `cargo test -p <crate>` is green. Record the delta in the task.

Existing benches: agent-core `decode`, `decode_concurrent`; agent `search`, `serve_events`, `persistence`.

---

## Bucket A — MECHANICAL (fix now, build+test only, no bench)
Grouped by disjoint file sets (no overlap with any measured file → safe to parallelize).

### MECH-1 — export/skills rendering  (files: `export.rs`, `skills.rs`)
- [T6-F6] `export.rs` pervasive `push_str(&format!())` → `write!(out, …)` (add `use std::fmt::Write`).
- [T6-F7] `skills.rs:1043` `html_escape`/`xml_escape` → return `Cow`, borrow when nothing to escape, bulk `push_str` spans.
- [T6-F8] `export.rs:974` `strip_control_chars` → return `Cow`, borrow when no control byte.
- [T8-F14] `skills.rs:917` `normalize_newlines` → `Cow`, early-return when no `\r`.
- [T8-F15] `skills.rs:1024` `format_available` (+ `agents::format_available` mirror) → `write!`.
- [T8-F16] `skills.rs:600` `sort_by_key` → `sort_by_cached_key`.

### MECH-2 — web tool  (files: `tools/web/extract.rs`, `tools/web/mod.rs`, `tools/web/ssrf.rs`)
- [T8-F2/F3] `extract.rs:51/47` `collapse_ws`/`text_of` → write tokens into one buffer, drop the `Vec<&str>`+intermediate `String`.
- [T8-F4] `extract.rs:70` `outline` → iterate `v.classes().take(2)` directly, drop the `Vec`.
- [T8-F5] `extract.rs:108` `locate` → collect borrowed `&str` into `String`, no per-child `String`.
- [T8-F7] `extract.rs:350` `clean_cell` → `Cow` early-return when no `\t\n\r`; push cells straight into `out`.
- [T8-F11] `mod.rs:308…341` `render_fetch` → `write!`.
- [T8-F12] `ssrf.rs:214` `SsrfResolver` → hold `Arc<EgressPolicy>`, clone the Arc; `host_allowed` allocation-free compare.

### MECH-3 — memory  (files: `memory/mod.rs`, `memory/file.rs`, `tools/memory.rs`)
- [T9-F5] `file.rs:195` `walk` → hoist `let base = self.dir()` out of the entry loop.
- [T9-F6] `file.rs:409` `search` → hoist the `format!` prefix; avoid per-line `to_lowercase`.
- [T9-F7] `mod.rs:147/195` `parse_in`/`classify` → `strip_prefix(root).and_then(|r| r.strip_prefix('/'))`.
- [T8-F18] `tools/memory.rs:138/155/127` render helpers → `write!`; `roots()` return array/iterator.

### MECH-4 — small tools + cold auth/settings  (files: `tools/find.rs`, `tools/ls.rs`, `tools/structured_output.rs`, `settings.rs`, `approval.rs`, `oauth/anthropic.rs`, `oauth/openai_codex.rs`)
- [T7-F10] `find.rs:244` → pre-size `out` with `with_capacity((paths*64).min(MAX_LISTING_BYTES))`.
- [T7-F8] `ls.rs:189` → fold the 3 `Vec` rebuilds into one pass / key-closure sort.
- [T8-F17] `structured_output.rs:204` → render before moving `input` into the slot; drop the clone.
- [T9-F4] `approval.rs:208` `SessionMemory` → single-`String` key + `&str` probe (no per-lookup tuple alloc).
- [T9-F13] `approval.rs:280` `truncate_value` → single `char_indices` pass.
- [T9-F9] `settings.rs:1130` → `char_indices` cursor, drop `Vec<char>`.
- [T9-F12] `oauth/{anthropic,openai_codex}.rs` → reuse a `OnceLock<Client>` in refresh/exchange. (github_copilot's Client handled with T9-F1.)

### Deferred-mechanical (done inside the owning file's measured pass, to keep baselines clean)
[T1-F7] agent.rs group key · [T3-F6] anthropic `normalize_tool_call_id` Cow · [T3-F7/F9] · [T4-F6] compaction `write!` · [T4-F10/F11/F12] · [T7-F5] edit `normalize_only` · [T7-F9] read resolve-path Cow · [T8-F19] subagent `describe` LazyLock · [T8-F13] `registry.names()` · [T2-F6/F7/F8/F9/F10/F11] · [T6-F5/F9/F10].

---

## Bucket B — MEASURED (bench before/after)

### M1 — `OutFrame::Raw(String)` → `Raw(bytes::Bytes)`  [T5-F1, T5-F3]  ·  bench: `serve_events`
Kills the per-event turn-recording clone + per-extra-sink fan-out copy; enables zero-copy WS. Metric: allocs/event, ns/event at 1 and 3 attached sinks. May extend `serve_events` to record + multi-sink.

### M2 — Delete the `out_tx` writer-task hop  [T5-F2]  ·  bench: `serve_events`
Broadcast directly under the fanout lock. Metric: allocs/event, ns/event. Correctness: keep background-login sends under the same lock.

### M3 — SSE `LineFramer` chunk-as-`Bytes` redesign  [T2-F1] (+ [T2-F10])  ·  bench: `decode`
Metric: allocs/event, bytes copied/event, ns/event. Watch line-straddling-chunk boundary case.

### M4 — SSE event buffer reuse  [T3-F1]  ·  bench: `decode`
Reusable cleared `String` + spillover `Vec`. Metric: allocs/event, ns/event.

### M5 — Chat-Completions tool-arg delta fast path  [T3-F5]  ·  bench: `decode` (extend w/ tool-arg stream)
Typed single-open-call fast struct. Metric: allocs/event on a tool-calling stream.

### M6 — Dialect encode: fuse passes / drop `Value`-tree intermediates  [T3-F2, T3-F3, T3-F4, T3-F7, T3-F8, T3-F9]  ·  bench: new `encode` bench (build_body over a synthetic N-turn history)
Metric: allocs/request, ns/request vs history length.

### M7 — Codex WS: stop deep-cloning the transcript  [T4-F1, T4-F2, T4-F3, T4-F4, T4-F11]  ·  bench: new `codex_ws` bench
`body_fingerprint`/`input_items`/`build_wire_body`/`wire_frame` borrow; frame parse from `&str`/slice. Metric: allocs/turn, bytes cloned/turn, ns/turn vs transcript size.

### M8 — `ContentBlock` payloads → `Arc<str>` / `Arc<[u8]>`  [T4-F8]  ·  bench: new `session_push` bench (make_mut on a shared history w/ large blocks)
Metric: bytes cloned + ns per `push`/`scrub` when the Arc is shared. Structural — lands before M7/agent.rs clones benefit.

### M9 — Tool-input take-semantics + `coerce_object_properties`  [T1-F2, T1-F4, T1-F5, T4-F5]  ·  bench: new `dispatch` micro-bench (coerce + gate a large write arg)
Metric: allocs/tool-call, bytes copied for a large file arg, ns/call.

### M10 — Per-turn `ModelRequest`/system-prompt clones  [T1-F1, T1-F3, T1-F6, T1-F7, T1-F8]  ·  bench: reuse `dispatch`/new `turn` micro-bench
Move `req`; `system: Option<Arc<str>>`; capture `&registry`; single tool-resolution pass. Metric: allocs/turn, ns/turn.

### M11 — `edit` CRLF/UTF-8 no-op path  [T7-F1, T7-F2, T7-F5, T7-F7]  ·  bench: new `edit` bench (4 MB pure-LF file — the doc's cited 73 ms case)
`Cow::Borrowed` + raw offsets when no `\r`. Metric: ms/edit, allocs/edit, bytes copied. Correctness: fuzzy/NFKC + CRLF files must stay green.

### M12 — Listing scan: kill the double deep-clone  [T6-F1, T6-F2, T6-F3, T6-F5]  ·  bench: `persistence`
Don't rebuild `fresh` from hits; borrowing `ListingIndexRef`; append chars into `search_text`; thread mtime. Metric: allocs/listing, ns/listing (warm + cold-miss), for a large history.

### M13 — `search_rank` allocation-free case-insensitive match  [T6-F4]  ·  bench: `search`
Metric: allocs/query, ns/query over N sessions incl. 50 KB `search_text`.

### M14 — Per-request capability/model-id caching  [T2-F2, T2-F3]  ·  bench: new `caps` micro-bench
Lowercase once, compute `ModelCaps` once, thread the `Copy` struct. Metric: allocs/request, ns/request.

### M15 — Hot-path I/O caching  [T8-F1, T9-F1, T9-F2, T9-F3, T8-F9, T8-F10]  ·  bench/measure: micro-bench `dynamic_footer` (ns/call) + syscall counts; child-spawn setup timing
`/etc/localtime` TZif → `OnceLock`; copilot routing memoized on `expires_at_ms`; shared `ModelOverrides`/`AuthStore` snapshot; per-dispatch context/mount reuse. Metric: syscalls/turn, ns/call.

### M16 — Blocking I/O off the reactor  [T7-F3, T7-F4, T7-F6, T9-F10]  ·  measure: latency test (concurrent progress during a large read) + allocs for T7-F4/F6
`read` text path → `spawn_blocking`; skip discarded `bash` capture; smaller snapshot decode; `mcp_auth_store` → `spawn_blocking`. Metric: reactor-stall latency, allocs/call.

### M17 — `web` extract row model → positional `Vec<String>`  [T8-F6, T8-F8]  ·  bench: new `web_extract` bench
Drop per-row `HashMap` + per-field name clones. Metric: allocs/row, ns over a large table. (Touches `whereexpr` interface — measured, not mechanical.)
