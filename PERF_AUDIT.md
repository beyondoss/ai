# Wasted-Work Audit — `crates/agent` + `crates/agent-core`

**Method.** Both crates (~113k LOC) were partitioned into 9 tracks, each audited in full by a dedicated pass hunting *only* for wasted work: unnecessary allocations, memory copies, threads/tasks, and redundant CPU/I-O. Findings below are deduplicated, cross-referenced, and ranked by **impact = frequency × cost**. Each carries a stable ID `[T<track>-F<n>]` back to its source pass.

**Headline.** The code is already unusually allocation-conscious (Arc-shared message vecs, borrowing serializers, sidecar listing caches, `bytes`/`memchr` framing). The real waste clusters in a handful of *systemic* patterns that each recur across many files. Fixing the ~8 cross-cutting themes below eliminates the large majority of the 101 findings.

Frequency legend (hottest first): **per-stream-event** → **per-turn** → **per-tool-call / per-request** → **per-listing/per-export** → **startup/cold**.

---

## Cross-cutting themes (fix these first — highest leverage)

### Θ1 — Per-stream-event payload copies (the hottest bytes in the system)
Every streamed token currently pays one or more heap allocations + memcpys.
- **[T2-F1]** `client.rs:831/924` — `LineFramer::extend` does `buf.extend_from_slice(chunk)`, memcpying **the entire response body** across the stream, even though line-splitting is already zero-copy via `split_to`. Redesign the framer to hold the current chunk as `Bytes` and `split_to` lines directly, carrying only the small unterminated remainder forward. *Eliminates ~1 full copy of the whole streamed body per turn.* (medium; bench with `benches/decode.rs`)
- **[T3-F1]** `dialect/mod.rs:646/669` — SSE event buffer does `data.push(payload.to_string())` + `mem::take` per event → one `String` **and** one `Vec` alloc/free per token delta. Keep a reusable cleared `String` buffer + spillover `Vec` only when a 2nd `data:` line actually arrives. (high)
- **[T4-F2]** `codex_websocket.rs:634` — every inbound WS frame is copied via `t.as_str().to_string()` / `b.to_vec()` before `from_str`. Use `serde_json::from_str(t.as_str())` / `from_slice(&b)` directly; delete the `text` binding. (high)
- **[T5-F1]+[T5-F3]** `serve.rs:8048/8058` — `OutFrame::Raw(String)` is `clone()`d for the in-flight turn recording **on every event** (even with zero connections attached) and again once per extra sink in multi-attach fan-out. **Change `OutFrame::Raw(String)` → `Raw(bytes::Bytes)`**: both copies become refcount bumps, and the WS `send_task` can hand `Bytes` straight to a zero-copy `Message::text`. One type change kills two findings and enables end-to-end zero-copy sends. (high)
- **[T4-F9]** `tool.rs:118` — `ToolProgress::emit` clones `id`+`name` `String`s on every progress snapshot (streaming `bash` emits many/turn). Store them as `Arc<str>`; clone the pointer. (medium)
- **[T3-F5]** `dialect/openai.rs:928/966` — OpenAI Chat-Completions tool-argument deltas always miss the typed fast path (`deny_unknown_fields`) and pay a full `serde_json::Value` parse per fragment. Add a typed fast-path struct for the single-open-tool-call steady state. (medium)

### Θ2 — Per-turn deep-clones of the whole transcript
The Codex delta transport exists *specifically to avoid moving the transcript each turn* — and then deep-clones it 3–4× anyway.
- **[T4-F1]** `codex_websocket.rs:187` — `body_fingerprint` does `body.clone()` (deep-clones the full `input` transcript array) then immediately `obj.remove("input")`. Clone only the retained (tiny) fields. Called ~2×/turn. (high)
- **[T4-F3]** `codex_websocket.rs:199/845/262/928` — the full `input` array is cloned 3–4×/turn via `input_items` for what are only length checks, prefix compares, and one `full_input.clone()` in a terminal `return` branch. Borrow the array; clone only the delta tail; move `full_input` instead of cloning. (high)
- **[T4-F4]** `codex_websocket.rs:270/285` — `build_wire_body` clones the whole body (incl. input) then overwrites `input`; `wire_frame` then clones the map a *second* time to add `"type"`. Clone object-minus-input; make `wire_frame` take `Value` by value and move the map. (high)
- **[T1-F3]** `agent.rs:2053/2107` — two speculative full `ModelRequest` clones per turn (retry snapshot + `before_hook` rollback) that the common no-retry/no-hook path never uses; each copies the multi-KB system prompt. Move `req` into the first attempt; skip `before_hook` when hooks are the default no-op. (medium)
- **[T1-F6]** `agent.rs:902` — static system prompt `String` re-cloned into the request every turn though it never changes. Make `ModelRequest.system` an `Option<Arc<str>>`; store as `Arc<str>` on the agent. (medium)
- **[T2-F4]** `client.rs:667` — full request-body `Value` (whole history) deep-cloned as a panic-rollback snapshot before the payload hook, on every request when any hook is installed. (medium)
- **[T2-F5]** `client.rs:702` — full body `Value` cloned for the Codex WS attempt while still holding the original for the HTTP fallback; the success path drops the original unused. Pass by move / return it back. (medium, Codex-only)

### Θ3 — Per-tool-call clones of coerced tool input (file bodies)
A `write`/`edit` argument *is* the whole file; it's currently copied several times before the tool runs.
- **[T1-F2]** `agent.rs:1580` — `outcomes[i].clone()` deep-copies the coerced arguments `Value` (entire file body) per call because the vec is shared behind one `Arc`. Give slots single-owner take semantics (`Mutex::take` or per-group owned sub-vecs) so the `Value` is **moved**. (high)
- **[T1-F4]** `agent.rs:1126/1460` — the tool-call `Value` is cloned into `calls`, then cloned **again** for `coerce_tool_arguments` (which takes `Value` by value). Change coercion to take `&Value` and clone only on the branch that rewrites a field. (medium)
- **[T4-F5]** `validation.rs:175` — `coerce_object_properties` does `remove` + `key.clone()` + `insert` per property on **every** tool dispatch. Use `get_mut` + `Value::take` — no key alloc, one lookup. (high)
- **[T1-F5]** `agent.rs:1461` — `tool.input_schema()` rebuilds the full JSON-schema `Value` on every tool call (schema is invariant). Cache it (`OnceCell<Value>` / precomputed map). (medium)
- **[T1-F1]** `agent.rs:1554` — the whole `ToolRegistry` HashMap (every `String` key + `Arc`) is cloned into each tool-call group though the group already borrows `&this` and never mutates the registry. Capture `&current_tools`. (medium)

### Θ4 — Make large immutable content payloads `Arc<str>` (structural, high-leverage)
- **[T4-F8]** `message.rs:33`, `session.rs:104/157` — every `ContentBlock` payload (`text`, `thinking`, base64 `ImageSource.data`, tool-result `content`) is an owned `String`. `Session::push`/`scrub` mutate via `Arc::make_mut`, which deep-clones the **entire** `Vec<Message>` — every string in every message, including megabyte base64 images — whenever a `ModelRequest` still shares the `Arc`. Make the big immutable fields `Arc<str>`/`Arc<[u8]>` so each per-message deep clone becomes a refcount bump. Serde round-trips unchanged. This is the single highest-leverage *type* change and it also cheapens Θ2/Θ3. (medium)

### Θ5 — `out.push_str(&format!(...))` everywhere (one idiom, one fix)
Each occurrence heap-allocates a throwaway `String` per fragment. Fix uniformly: `use std::fmt::Write; write!(out, …)` straight into the buffer.
- **[T6-F6]** `export.rs` — pervasive (lines 109,116,120,254,266,280,293,299,340,377,612,729,781,790,812,887,1001,1023,…), across the whole transcript + every abandoned branch. (Note: current `use std::io::Write` does **not** enable `write!` on `String`.) (high)
- **[T4-F6]** `compaction.rs:633/647/651` — `render_prefix_impl` per content block in the summarized prefix. (high)
- **[T8-F11]** `web/mod.rs:308/314/317/341` — `render_fetch`, once per response header. (high)
- **[T8-F15]** `skills.rs:1024` (+ `agents::format_available`) — 3× per visible skill, every prompt build incl. each child. (high)
- **[T8-F18]** `memory.rs:138/155/127` — per listing entry / per search hit. (high)

### Θ6 — Allocate-on-no-op: `replace` / `to_lowercase` / CRLF-normalize / escape that copy even when nothing changes
Gate each on a cheap scan and return `Cow::Borrowed` on the common (unchanged) path.
- **[T7-F1]** `edit.rs:175` — `body.replace("\r\n","").contains('\n')` copies the **entire file** to compute one bool. Scan bytes once instead. (high)
- **[T7-F2]** `edit.rs:177/649` — `strip_crlf_with_map` unconditionally builds a full copy + a **4×-file-size** `Vec<u32>` offset map + a redundant `String::from_utf8` re-validation, even for pure-LF files (the common case, where it's a no-op). Gate on `body.contains('\r')`; use `Cow::Borrowed` + raw offsets when absent. ~5× file-size of alloc avoided per edit. (high)
- **[T7-F7]** `edit.rs:199` — `old`/`new` `replace("\r\n","\n")` allocates even when no CRLF present. (high)
- **[T8-F14]** `skills.rs:917` — `normalize_newlines` makes two full-string `replace` passes unconditionally; LF-only input allocates two identical copies. Early-return `Cow::Borrowed` when no `\r`. (high)
- **[T6-F4]** `session_store.rs:227` — `search_rank` does `field.to_lowercase()` per field including the **50 KB** `search_text`, per session, per query → O(sessions × 50 KB) allocate-and-discard. Allocation-free case-insensitive substring search. (high)
- **[T6-F7]** `skills.rs:1043` (`html_escape`/`xml_escape`, used across `export.rs`) — always allocates a full copy and pushes char-by-char even when nothing needs escaping. Return `Cow`; bulk-`push_str` unescaped spans. (medium)
- **[T6-F8]** `export.rs:974` — `strip_control_chars` always copies every (large) tool-result body to remove control bytes that are essentially never present. Return `Cow`. (medium)
- **[T8-F7]** `web/extract.rs:350` — `clean_cell` `replace(['\t','\n','\r'], " ")` allocates per cell unconditionally. `Cow` early-return. (high)
- **[T9-F6]** `memory/file.rs:409` — `line.to_lowercase()` per line for a `contains` test. (high, cold-ish)

### Θ7 — Per-turn / per-request / per-spawn redundant I/O
State that is constant for the process (or the session) is re-read and re-parsed from disk on hot paths.
- **[T8-F1]** `resources.rs:342→491→546` — `dynamic_footer` reads **`/etc/localtime` and parses the whole TZif blob on every turn** to recompute an effectively-constant UTC offset. Cache in `OnceLock` (optionally keyed on `TZ`). (high)
- **[T9-F1]** `oauth/github_copilot.rs:469` — Copilot `credential()` re-reads + re-parses **`auth.json` on every model request** (defeating the `OAuthCredentialSource` in-memory cache) to derive routing that only changes on token refresh. Memoize keyed on `expires_at_ms`. (high)
- **[T9-F2]** `gateway_credential.rs:248` — `resolve_gateway_credential` re-opens+parses **both `auth.json` and `models.json`** on every client build (per subagent spawn, per `serve` model-switch/fork/clone). Load once at startup; thread the parsed value in. (high)
- **[T9-F3]** `main.rs:109` — `model_override_extra_headers` re-parses `models.json` a **second time** in the same client build, µs after F2. Reuse the one parse. (high)
- **[T8-F9]** `subagent.rs:698→resources.rs:232` — every subagent child re-walks `cwd`→root reading `AGENTS.md`/`CLAUDE.md` and re-renders the skills block, though up-to-8 siblings share the same cwd/tool-set. Compute context-files + skills block once per dispatch. (high)
- **[T8-F10]** `subagent.rs:697` — `mount_sections` re-reads each `MEMORY.md` index once per child in a fan-out; all siblings share the same snapshot. Read once in dispatch. (medium)
- **[T2-F2]+[T2-F3]** `models.rs` route-capabilities chain + `client.rs:522` — the model id is `to_ascii_lowercase()`'d **4+ times per request**, the whole family-branch scan runs repeatedly, and `capabilities_impl` runs twice per request. `ModelCaps` is `Copy`: lowercase once, compute caps once, thread the struct. (high)

### Θ8 — Blocking I/O on the async reactor (thread-starvation risk)
`serve_ws` runs a per-session **current-thread** runtime; inline blocking work pins it (this is exactly why `edit` was moved to `spawn_blocking`).
- **[T7-F3]** `read.rs:326` — `read`'s text path does `std::fs::File::open` + `BufReader` line reads **inline on the reactor**, including a whole-file drain after truncation just to count total lines. Its siblings `grep`/`find`/`edit`/image-`read` all use `spawn_blocking`; the text path is the gap. (medium)
- **[T9-F10]** `mcp_auth_store.rs:83/234` — async `load`/`save`/`clear` call sync `fs` directly and can `std::thread::sleep` up to 5 s under lock contention, on a tokio worker. Wrap in `spawn_blocking` like `auth_credential_source.rs`. (medium)

---

## Remaining findings (by track)

### Track 1 — `agent-core/agent.rs` (core loop)
- **[T1-F7]** `agent.rs:1347` — `target.clone()` + `format!` to build the group key; build from `target.as_deref()`. (high)
- **[T1-F8]** `agent.rs:1332–1357` — 3–4 separate `calls.iter()` passes + repeated `ToolRegistry::get` (each an `Arc` bump) per call; resolve each tool once into a parallel vec and derive everything in one pass. (medium)
- **[T1-F9]** `agent.rs:1962` — fresh unbounded mpsc channel allocated per call in the serial interleaved path; hoist one channel, clone the sender. (low)
- *Cleared as non-waste:* `session.messages.clone()`, `current_tool_defs.clone()`, `req.clone()` messages field (all `Arc` bumps); `Accumulator::apply` borrows; `buffer_unordered` (no per-item task spawn); no `block_on`/`spawn_blocking` misuse.

### Track 2 — `models.rs` / `client.rs` / `transport.rs`
- **[T2-F6]** `client.rs:793` — response headers cloned into `Vec<(String,String)>` for a read-only hook; pass borrowed `&str` pairs. (medium)
- **[T2-F7]+[T2-F11]** `client.rs:1133/1085` — `anthropic-beta` Vec+`join` and full-body `.json(body)` re-serialization recomputed **inside** the retry loop though invariant across attempts; hoist before the loop, pre-serialize body to `Bytes`. (high that it recomputes; low impact — retry-only)
- **[T2-F8]** `client.rs:565` — `static_headers`/`auth_header`/`auth_header_prefix` cloned out of an owned credential; destructure and move. (medium)
- **[T2-F9]** `models.rs:3058` — `available_thinking_levels` heap-allocs a `Vec` of ≤6 one-byte enums; `ArrayVec` or compute next level directly. (cold)
- **[T2-F10]** `client.rs:938` — `find_line_break` re-scans the buffered partial-line prefix each chunk (O(n²) for a long line across many chunks); track a `scanned_upto` offset. (low; superseded by T2-F1)
- *Cleared:* message-repair returns `Cow`; `req.tools`/`http`/`extra_headers` clones are `Arc` bumps; `reqwest::Client` built once.

### Track 3 — dialects
- **[T3-F2]** `anthropic.rs:132` — `build_body` serializes the whole history to a `Value` tree then walks it 4–6× in separate passes (each a full `as_array_mut` scan) even when the common case mutates nothing. Fuse into one `type`-keyed pass; short-circuit no-op passes with a flag. (medium)
- **[T3-F3]** `openai.rs:518` / `openai_responses.rs:450` / `anthropic.rs:98` — all three `build_body`s materialize the whole request as a `serde_json::Value` tree that the caller immediately serializes to bytes; for the two OpenAI dialects (never mutated post-build) serialize a borrowed typed view straight to the writer. (medium)
- **[T3-F4]+[T3-F8]** `openai_responses.rs:363` / `openai.rs:83` — stored reasoning signatures / Gemini `thought_signature` are `from_str::<Value>`-parsed then re-serialized (string→tree→string round-trip) per request, per historical item. Use `Box<RawValue>` passthrough. (medium)
- **[T3-F6]** `anthropic.rs:476` — `normalize_tool_call_id` always builds a fresh `String` even when the id is already conformant (common). Return `Cow`; scan first. (high, cross-model-only)
- **[T3-F7]** `openai.rs:441` — each assistant message walked 4× (text / tool_calls / reasoning / reasoning_details); single pass. (medium)
- **[T3-F9]** `openai.rs:185` — `has_tool_history` does a full extra O(history) scan on the no-tools path; track the flag in the main encode loop. (medium)
- *Cleared:* `repair_orphaned_tool_use`/`ensure_non_empty_content` use `Cow::Borrowed` fast paths; fast-path decoders move owned `String`s (no extra copy).

### Track 4 — compaction / steering / codex_websocket / core types
- **[T4-F7]** `tool.rs:332/295` — `ToolRegistry::definitions()` rebuilds+clones every tool's name/description/`input_schema()` (fresh `json!`) per call; cache the sorted `Vec<ToolDef>`, invalidate on register/retain. (medium — impact depends on call frequency, see T1-F5)
- **[T4-F10]** `compaction.rs:664` — `truncate_chars` scans the string up to 3× (`chars().count()` twice + `take`) and clones on the no-op path; single `char_indices().nth(max)` pass, push borrowed slice. (medium)
- **[T4-F11]** `codex_websocket.rs:666` — harvested output items cloned though the source `Value` is dropped; reorder to `Value::take` the item after decode. (medium)
- **[T4-F12]** `compaction.rs:104/695` — `merge_provenance`/`extract_file_ops` dedupe with O(n²) `Vec::contains` + upfront list clones; `HashSet` seen-guard only if lists can grow. (low)
- *Cleared:* `write_lock.rs` critical sections short, no lock-across-await; `steering.rs` mutexes cold; `count_value_chars` avoids materializing JSON.

### Track 5 — serve / serve_ws
- **[T5-F2]** `serve.rs:2446` — redundant per-event unbounded-mpsc hop + dedicated writer task between the (synchronous) event sink and the synchronous `broadcast`; the sink can call `broadcast` directly under the fanout lock. Deletes a node-alloc + task round-trip per event. (medium — must keep the background login task's sends under the same lock)
- **[T5-F4]** `serve_ws.rs:351` — inbound WS command copied via `text.as_str().to_owned()` (up to 8 MiB for a pasted `prompt`); `String::from(text)` reuses the uniquely-owned `Utf8Bytes` buffer. (medium)
- **[T5-F5]** `serve.rs:7939` — `set_history` does `Arc::new(ids.to_vec())` per committed change (gated on pointer change; already acknowledged in-code). (low)
- **[T5-F6]** `serve_ws.rs:328` — `text.contains("list_daemon_sessions")` prefilter scans every inbound message; **effectively a non-finding** (cheaper than always parsing JSON). (—)
- *Cleared:* thread-per-session + current-thread runtime justified (non-`Send` sink); all `spawn_blocking` justified; single-sink `broadcast` already zero-copy; `event_frame` single-pass serialize.

### Track 6 — session_store / export
- **[T6-F1]** `session_store.rs:3660` — warm listing **double-deep-clones** every session's `SessionMeta` (incl. up-to-50 KB `search_text`) twice per call — once via `to_meta()`, once via `ListingIndexEntry::new` — and the `fresh` map is usually discarded (the `unchanged` check skips the write). Don't rebuild `fresh` from hits; write the index only on misses/deletions. (high)
- **[T6-F2]** `session_store.rs:3595` — `store_listing_index` clones the entire index map just to wrap it with a `version` field for serialization; use a borrowing `ListingIndexRef<'a>`. (high)
- **[T6-F3]** `session_store.rs:3853/3931` — cache-miss scan builds a throwaway `Vec<&str>`+`join` per message, then a second `String` (`taken`) + a re-count, before `push_str`. Append text-block chars straight into `search_text` up to the budget. (high)
- **[T6-F5]** `session_store.rs:3659/3785` — cache-miss files are `stat`'d twice (`file_stamp` then `read_listing`'s `mtime_secs`); thread the mtime through. (medium)
- **[T6-F9]** `export.rs:181` — `render_html_inner` clones every trailing event into a fresh `Vec` (incl. `Custom{data:Value}`) for a read-only renderer; collect `Vec<&ExportEvent>`. (medium)
- **[T6-F10]** `session_store.rs:1374/1599` — `rewrite`/`compact` call `now_secs()` + `new_id()` (2 clock syscalls) per message; hoist one `now_secs()` per rewrite. (medium, rare)
- **[T6-F11]** `session_store.rs:1219` — `append_new` clones each new `Message` into its `Node` — **forced** by the ownership split; not actionable without restructuring `Session` to `Arc<Message>` per element. (—)
- *Cleared:* per-turn append uses borrowing `EntryRef` → one buffered `write_all`; listing scan is a true streaming parse; compaction shares kept suffix via `Arc::clone`.

### Track 7 — file tools
- **[T7-F4]** `exec.rs:180` — `RealRunner` maintains ≤256 KiB head/tail `Capture` and produces `from_utf8_lossy(...).into_owned()` `stdout`/`stderr` **that streaming `bash` discards** (its sink already got every chunk). Skip the capture when `on_chunk.is_some()`. (medium)
- **[T7-F5]** `edit.rs:381` — `find_spans` builds the `old_string` offset map via `normalize_with_map(old).0` then throws the map away; use the existing `normalize_only(old)`. (high)
- **[T7-F6]** `output.rs:514` — `snapshot_text` decodes+copies the whole ~100 KiB rolling tail on every ~100 ms throttle tick, most of which `truncate_tail` discards; decode only the trailing `max_bytes`. (medium)
- **[T7-F8]** `ls.rs:189` — entries `Vec` rebuilt 3× (add key → display); fold into one pass / sort with a key closure. (medium, bounded)
- **[T7-F9]** `read.rs:130/230` — `resolve_read_path` returns a second `path.to_string()` on the exists fast path; take/return `String` or `Cow`. (high, tiny)
- **[T7-F10]** `find.rs:244` — output buffer `String::new()` not pre-sized, unlike `grep`/`read`; `with_capacity((paths.len()*64).min(MAX_LISTING_BYTES))`. (high, low impact)
- *Cleared:* `grep`/`find`/image-`read`/`edit` `spawn_blocking` justified; `bash` `clean`/`clean_str` borrow via `Cow`; no `block_on` misuse.

### Track 8 — web / subagent / mcp / structured_output / skills / resources / prompts
- **[T8-F2]+[T8-F3]** `web/extract.rs:51/47` — `collapse_ws` builds a throwaway `Vec<&str>` per call (per extracted field / table cell / located element), and `text_of` allocates 3× for one collapsed string. Write tokens directly into one output buffer. (high)
- **[T8-F4]** `web/extract.rs:70` — `outline` collects classes into a `Vec<&str>` (≤2) just to iterate it, per element in the document. Iterate the classes iterator. (high)
- **[T8-F5]** `web/extract.rs:108` — `locate` allocates a `String` per text child before collecting into `String`, per element. Collect borrowed `&str` directly. (high)
- **[T8-F6]+[T8-F8]** `web/extract.rs:217/299` — extract/table rows are `HashMap<String,String>` with the field/header name **cloned per row per field** (N×M clones + hashing); store rows as positional `Vec<String>` aligned with the fixed field list. (medium)
- **[T8-F12]** `web/ssrf.rs:214` — `SsrfResolver::resolve` deep-clones the whole `EgressPolicy` (incl. `allow_hosts` Vec) per connection + per redirect hop; store `Arc<EgressPolicy>`. (`host_allowed` also lowercases per check.) (high)
- **[T8-F13]** `resources.rs:39` — `default_system_prompt` calls `registry.definitions()` (materializes every tool's full `input_schema()` `Value`) just to read names; add/use `registry.names()`. (medium)
- **[T8-F16]** `skills.rs:600` — `sort_by_key(|p| p.components().count())` recomputes the key O(n log n)×; `sort_by_cached_key`. (startup)
- **[T8-F17]** `structured_output.rs:204` — `slot.set(input.clone())` before rendering the (potentially large) payload; render first, then move `input`. (high)
- **[T8-F19]** `subagent.rs:170` — `describe()` `format!`s a static description (only `const`s) per `Subagent` instance; `LazyLock<String>`. (high, small)
- *Cleared:* `whereexpr::Filter` compiles regex once; `mcp.rs` connects at startup via `join_all`; subagents run as in-process `buffered` futures (no per-child thread/`block_on`); skill/prompt discovery is startup-only, `Arc`-shared.

### Track 9 — main / settings / stores / oauth
- **[T9-F4]** `approval.rs:208` — `SessionMemory::lookup` builds an owned `(String,String)` tuple just to **probe** the map, per gated call (with `--approve`); `remember` allocs 2 more. Single-`String` key + `&str` probe, or `Borrow` wrapper. (medium)
- **[T9-F5]** `memory/file.rs:195` — `walk` calls `self.dir()` (clones a `PathBuf`, and takes an RwLock read for `Shared`) **per directory entry**; hoist `let base = self.dir()` before the loop. (high)
- **[T9-F7]** `memory/mod.rs:147/195` — `parse_in`/`classify` build `format!("{root}/")` prefixes per parse; use `strip_prefix(root).and_then(|r| r.strip_prefix('/'))`. (medium)
- **[T9-F8]** `settings.rs:290` — `merge_over` deep-clones ~25 fields of **both** configs; `project` is owned and could be moved. (startup)
- **[T9-F9]** `settings.rs:1130` — `resolve_config_value_template` collects the whole value into `Vec<char>`; use a `char_indices` cursor. (low)
- **[T9-F11]** `auth_credential_source.rs:82` — cached OAuth token `String`-cloned on every cache-hit request; store `Arc<str>`. (medium, tokens short)
- **[T9-F12]** oauth `anthropic.rs:102`, `openai_codex.rs:175`, `github_copilot.rs:205` — `reqwest::Client::new()` rebuilt per refresh/exchange; module-level `OnceLock<Client>`. (low, ~once per token lifetime)
- **[T9-F13]** `approval.rs:280` — `truncate_value` scans with `chars().count()` up to 3×; single `char_indices` pass. (low, human-gated)
- **[T9-F14]** `main.rs:3853` — `SubagentCtx` clones `skills`/`agent_defs`/`mounts` (and re-clones `mounts`); `Arc`-wrap once up front and borrow through. (startup)
- *Cleared:* `policy.rs` (deny-lists lowercased + globs compiled once at construction); `retry.rs` (`OnceLock` entropy salt, atomic counter, arithmetic backoff); `auth_store.rs` double-checked locking + `spawn_blocking` bridge intentional; `oauth/callback_server.rs` thread on `spawn_blocking` justified.

---

## Suggested execution order

1. **`OutFrame::Raw(String)` → `Bytes`** [Θ1: T5-F1/F3] — one type change, kills the always-on per-event server copy + enables zero-copy WS.
2. **Codex WS transcript clones** [Θ2: T4-F1/F3/F4] + **frame→String copy** [T4-F2] — restores the delta transport's whole reason to exist.
3. **`Arc<str>` for `ContentBlock` payloads** [Θ4: T4-F8] — structural; cheapens make-mut clones and Θ2/Θ3 downstream.
4. **SSE framer + event buffer** [Θ1: T2-F1, T3-F1] — the streamed-token hot path (bench with `benches/decode.rs`).
5. **Coerced-input take-semantics** [Θ3: T1-F2/F4, T4-F5] — stop copying file bodies per tool call.
6. **`edit` CRLF/UTF-8 no-op path** [Θ6: T7-F1/F2/F5/F7] — the acknowledged reactor-staller.
7. **Per-turn/per-request I/O caching** [Θ7: T8-F1, T9-F1/F2/F3, T2-F2] — cache what's process-constant.
8. **`write!`-into-buffer sweep** [Θ5] + the `Cow` no-op-escape sweep [Θ6] — mechanical, wide.
9. **Listing double-clone** [T6-F1/F2/F3] — unbounded-growth path.
10. **Blocking-I/O-on-reactor** [Θ8: T7-F3, T9-F10].

*All findings carry file:line; each is independently verifiable before changing.*
