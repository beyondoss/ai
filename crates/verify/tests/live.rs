//! Live cells: real clients, through a gateway built from this commit, to real providers.
//!
//! Each cell is one test, named `CLAIMS::client::route::probe` (claims joined by `+`), so nextest
//! filters, isolates and reports them, and `verify status` reads their outcomes from the JUnit
//! report. A cell boots its own gateway and nats-server with the real pool keys from `.env`, runs
//! one probe in a pinned client (`verify/clients/`), and then checks two witnesses:
//!
//! 1. The client's own verdict: the probe parsed everything and the task's structure held.
//! 2. The gateway's ledger: for every HTTP call the client made, exactly one `ai.usage` row with
//!    that `x-beyond-request-id`, whose tokens equal what the client was shown (normalized for the
//!    wire), served by the provider the route expects. A coding agent's calls aren't visible to
//!    us, so its ledger is checked in aggregate.
//!
//! An E7 cell (a coding agent) checks what the harness itself displays instead of its task: the
//! models it lists against /v1/models, and the session cost it shows against the ledger priced at
//! the card ([`e7_problems`]).
//!
//! Cells are listed only with `VERIFY_LIVE=1`, so an ordinary test run never spends money. A cell
//! whose key or client is missing is not listed at all; `verify status` reports that client as
//! missing for the claim (PARTIAL), never as a pass.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/live.rs"]
mod common;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::free_port;
use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::Value;

/// The dev signing key (seed `[7; 32]`, kid 1) and the tenant-1 token minted from it; the same
/// constants `mise run ai:mint-dev-key` prints.
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// Where a cell sends its traffic: a catalog row and the pool keys the gateway holds.
#[derive(Clone, Copy)]
struct Route {
    name: &'static str,
    model: &'static str,
    /// `(provider, env var)` pool keys; every one must be set for the cell to be listed.
    pools: &'static [(&'static str, &'static str)],
    /// Providers pointed at a dead port, to force failover.
    dead: &'static [&'static str],
    /// The provider every billing row must name.
    serves: &'static str,
}

const CLAUDE: Route = Route {
    name: "claude",
    model: "claude-haiku-4-5",
    pools: &[("anthropic", "ANTHROPIC_API_KEY")],
    dead: &[],
    serves: "anthropic",
};
/// The GPT row: a reasoning model whose primary is OpenAI's own Chat Completions (Chat clients are
/// relayed, Messages clients translated onto it), with a Responses arm. gpt-5.1 is the cheapest
/// such row OpenAI has not scheduled to retire (gpt-5-mini goes 2026-12-11; gpt-5.4 and later
/// reach OpenAI over Responses, D114). It reasons only when asked (effort `none` by default), so
/// a cell that doesn't ask pays for no reasoning.
const GPT: Route = Route {
    name: "gpt",
    model: "gpt-5.1",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};
/// A Codex-native model: Codex's request (hosted and namespace tools, `store: false`) relays
/// unchanged to OpenAI's Responses API, the first candidate of a Responses-first row.
const CODEX: Route = Route {
    name: "codex",
    model: "gpt-5.3-codex",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};
/// The Claude row with only an OpenRouter key: OpenRouter's Chat Completions serves it.
const OPENROUTER: Route = Route {
    name: "openrouter",
    model: "claude-haiku-4-5",
    pools: &[("openrouter", "OPENROUTER_API_KEY")],
    dead: &[],
    serves: "openrouter",
};
/// Anthropic unreachable: the Claude row must fail over to OpenRouter before the client notices.
const FAILOVER: Route = Route {
    name: "failover",
    model: "claude-haiku-4-5",
    pools: &[
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
    dead: &["anthropic"],
    serves: "openrouter",
};
/// The Claude row with Anthropic unkeyed: Amazon Bedrock serves (its key is minted per run from
/// the local AWS credential, see `verify/clients/py/bedrock_token.py`).
const BEDROCK: Route = Route {
    name: "bedrock",
    model: "claude-haiku-4-5",
    pools: &[("bedrock", "AWS_BEARER_TOKEN_BEDROCK")],
    dead: &[],
    serves: "bedrock",
};
/// xAI's own API: grok reports reasoning beside completion_tokens (D23).
const XAI: Route = Route {
    name: "xai",
    model: "grok-4.3",
    pools: &[("xai", "XAI_API_KEY")],
    dead: &[],
    serves: "xai",
};
/// Together's own API (OpenAI-compatible Chat): an open-weights row whose primary is Together.
const TOGETHER: Route = Route {
    name: "together",
    model: "llama-3.3-70b-versatile",
    pools: &[("together", "TOGETHER_API_KEY")],
    dead: &[],
    serves: "together",
};
/// The Claude row with two live pools, so routing has a real choice. The probe names the provider
/// each call must land on (steering headers, session pins), so `serves` is empty: any.
const POOLED: Route = Route {
    name: "pooled",
    model: "claude-haiku-4-5",
    pools: &[
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
    dead: &[],
    serves: "",
};
/// [`POOLED`] on Claude Sonnet 4.6, whose minimum cacheable prompt is below pi's whole prompt
/// (~2.5k; measured 2026-10-01: a 1,056-token system prompt was written to the cache); on Haiku
/// 4.5 (4096) pi's session never caches, so a pin can't show it. Sonnet 4.5, this route's row
/// until then, retires 2026-11-30; 4.6 is its price tier and still takes the
/// `thinking.type.enabled` pi sends, which Sonnet 5 and 5.5 refuse (400 "Use
/// thinking.type.adaptive", measured on this cell 2026-10-01).
const POOLED_SONNET: Route = Route {
    name: "pooled-sonnet",
    model: "claude-sonnet-4-6",
    pools: &[
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
    dead: &[],
    serves: "",
};
/// A Claude 5.x row, for behavior that differs on newer Claude (forced tools, effort).
const SONNET: Route = Route {
    name: "sonnet",
    model: "claude-sonnet-5-5",
    pools: &[("anthropic", "ANTHROPIC_API_KEY")],
    dead: &[],
    serves: "anthropic",
};
/// A 16k-output row, for max_tokens clamping, and the smallest window OpenAI serves on a row not
/// scheduled to retire (128k), for a context overflow (TRN-18) that is rejected before billing.
/// That cell was on gpt-4's 8k window until the gpt-4 row was removed (D243).
const GPT4O_MINI: Route = Route {
    name: "gpt4o-mini",
    model: "gpt-4o-mini",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};
const EMBED: Route = Route {
    name: "embed",
    model: "text-embedding-3-small",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};

#[derive(Clone, Copy)]
enum Runtime {
    Python,
    Node,
    /// A coding agent (Claude Code, Codex, opencode, pi) driven by `verify/clients/harness.py`.
    Harness,
    /// A coding agent behind `verify/clients/harness_long.py`'s recording proxy (`cell` mode), so
    /// each of its HTTP calls is held to the ledger ([`recorded_problems`]), not just the total.
    Recorded,
}

/// `(claims, client, runtime, probe, routes, extra claims on the failover route)`. One line per
/// client × probe; the routes expand it into cells.
type Cell = (
    &'static str,
    &'static str,
    Runtime,
    &'static str,
    &'static [Route],
    &'static str,
);

const GEN: &[Route] = &[CLAUDE, GPT, OPENROUTER, FAILOVER, BEDROCK, XAI, TOGETHER];
const SESSION: &[Route] = &[CLAUDE, GPT, OPENROUTER];
const ONE: &[Route] = &[GPT];
const CODEX_ROWS: &[Route] = &[CODEX, CLAUDE, OPENROUTER];
/// The three model families whose request mapping differs most: Claude, GPT and grok.
const FAMILIES: &[Route] = &[CLAUDE, GPT, XAI];
const CLAUDE_GPT: &[Route] = &[CLAUDE, GPT];

#[rustfmt::skip]
const CELLS: &[Cell] = &[
    // Python SDKs and frameworks. The basic cells carry the generic ledger claims (BIL-6/7/13):
    // every cell checks them, these name them across every translation pair.
    ("E1+B1+S1+S2+BIL-6+BIL-7+BIL-13", "openai-py", Runtime::Python, "chat_basic", GEN,     "R1"),
    ("E2+B1+S1+BIL-6+BIL-7+BIL-13", "anthropic-py", Runtime::Python, "messages_basic", GEN, "R1"),
    ("E3+TRN-1+CAT-9+B1", "openai-py",     Runtime::Python, "responses_basic",  SESSION,  ""),
    ("TRN-1+B1+BIL-6",    "openai-py",     Runtime::Python, "responses_basic",  &[FAILOVER, BEDROCK, XAI, TOGETHER], ""),
    ("E4",                "openai-py",     Runtime::Python, "models_list",      ONE,      ""),
    ("E4",                "anthropic-py",  Runtime::Python, "models_list",      ONE,      ""),
    ("T1+B1",             "openai-py",     Runtime::Python, "tools_chat",       GEN,      "R1"),
    ("T1+B1",             "anthropic-py",  Runtime::Python, "tools_messages",   GEN,      "R1"),
    ("M1+B1",             "openai-py",     Runtime::Python, "embeddings",       &[EMBED], ""),
    ("E1+T1+B1",          "langchain",     Runtime::Python, "langchain_chat",   GEN,      "R1"),
    ("E3+T1+B1+TRN-24",   "openai-agents", Runtime::Python, "agents_sdk",       SESSION,  ""),
    ("E1",                "openai-agents", Runtime::Python, "agents_chat",      SESSION,  ""),
    // Endpoint and billing detail (Python).
    ("E5+B4+BIL-1",       "openai-py",     Runtime::Python, "responses_count_compact", ONE, ""),
    ("E5+B4",             "anthropic-py",  Runtime::Python, "count_tokens",     &[CLAUDE], ""),
    ("T2+E2",             "anthropic-py",  Runtime::Python, "thinking_replay",  &[CLAUDE, BEDROCK, FAILOVER], ""),
    ("T2+BIL-9",          "openai-py",     Runtime::Python, "reasoning_replay", ONE,      ""),
    ("TRN-8+T2",          "openai-agents", Runtime::Python, "agents_reasoning", CLAUDE_GPT, ""),
    ("T3",                "openai-py",     Runtime::Python, "vision_chat",      CLAUDE_GPT, ""),
    ("T3",                "anthropic-py",  Runtime::Python, "vision_messages",  CLAUDE_GPT, ""),
    ("T4",                "openai-py",     Runtime::Python, "structured_chat",  &[CLAUDE, GPT, XAI, TOGETHER], ""),
    ("T4",                "anthropic-py",  Runtime::Python, "structured_messages", CLAUDE_GPT, ""),
    ("T4",                "langchain",     Runtime::Python, "langchain_structured", FAMILIES, ""),
    ("T5",                "openai-py",     Runtime::Python, "reasoning_effort", &[CLAUDE, SONNET, GPT, XAI], ""),
    ("T6",                "openai-py",     Runtime::Python, "typed_error",      FAMILIES, ""),
    ("T6",                "anthropic-py",  Runtime::Python, "typed_error",      FAMILIES, ""),
    ("R3",                "openai-py",     Runtime::Python, "steer_providers",  &[POOLED], ""),
    ("R4+B3",             "anthropic-py",  Runtime::Python, "session_pin",      &[POOLED], ""),
    ("R5",                "openai-py",     Runtime::Python, "big_body",         &[CLAUDE, GPT, FAILOVER], ""),
    ("R5",                "anthropic-py",  Runtime::Python, "big_body",         &[CLAUDE, FAILOVER], ""),
    ("B2+BIL-3+BIL-20",   "openai-py",     Runtime::Python, "stream_abort",     FAMILIES, ""),
    ("BIL-3",             "openai-py",     Runtime::Python, "cancel_before_head", CLAUDE_GPT, ""),
    ("BIL-2",             "openai-py",     Runtime::Python, "stream_no_usage",  &[CLAUDE, GPT, XAI, TOGETHER], ""),
    ("B3+BIL-8+BIL-11",   "anthropic-py",  Runtime::Python, "prompt_cache",     &[CLAUDE, BEDROCK], ""),
    ("K1+B3+BIL-8",       "openai-py",     Runtime::Python, "auto_cache",       &[CLAUDE, BEDROCK, GPT], ""),
    ("K1",                "langchain",     Runtime::Python, "langchain_cache",  &[CLAUDE], ""),
    ("A1",                "openai-py",     Runtime::Python, "byo_key",          ONE,      ""),
    ("A1",                "anthropic-py",  Runtime::Python, "byo_key",          &[CLAUDE], ""),
    ("BIL-1+BIL-7",       "openai-py",     Runtime::Python, "provider_routed",  &[GPT, XAI], ""),
    ("BIL-1+BIL-7",       "anthropic-py",  Runtime::Python, "provider_routed",  &[CLAUDE], ""),
    ("BIL-9",             "openai-py",     Runtime::Python, "reasoning_metered", &[XAI, GPT], ""),
    ("BIL-10",            "anthropic-py",  Runtime::Python, "web_search",       &[CLAUDE], ""),
    ("BIL-11+B3",         "anthropic-py",  Runtime::Python, "cache_ttl_1h",     &[CLAUDE], ""),
    ("TRN-2",             "openai-py",     Runtime::Python, "mid_system",       FAMILIES, ""),
    ("TRN-4+B3",          "anthropic-py",  Runtime::Python, "cache_control_turns", &[CLAUDE], ""),
    ("TRN-4+B3",          "openai-py",     Runtime::Python, "cache_control_parts", &[CLAUDE], ""),
    ("TRN-5",             "openai-py",     Runtime::Python, "max_tokens_clamp", &[GPT4O_MINI], ""),
    ("TRN-5",             "anthropic-py",  Runtime::Python, "max_tokens_clamp", &[GPT4O_MINI], ""),
    // TRN-6 is the translated path: Sonnet 5.5 itself 400s on a forced Messages tool_choice (relayed
    // as-is to a Messages client); a Chat named tool must still come back as a tool call.
    ("TRN-6+T1",          "openai-py",     Runtime::Python, "tools_chat",       &[SONNET], ""),
    ("TRN-11",            "anthropic-py",  Runtime::Python, "strict_tools",     &[CLAUDE], ""),
    ("TRN-11",            "openai-py",     Runtime::Python, "strict_tools",     &[CLAUDE], ""),
    ("TRN-15",            "openai-py",     Runtime::Python, "explicit_nulls",   GEN,      ""),
    ("TRN-16+TRN-2",      "openai-py",     Runtime::Python, "developer_role",   &[XAI, TOGETHER], ""),
    ("TRN-18",            "openai-py",     Runtime::Python, "context_overflow", &[GPT4O_MINI], ""),
    ("TRN-18",            "anthropic-py",  Runtime::Python, "context_overflow", &[GPT4O_MINI], ""),
    ("W5+TRN-24",         "openai-agents", Runtime::Python, "agents_handoff",   CLAUDE_GPT, ""),
    ("W6",                "langchain",     Runtime::Python, "langchain_agent",  CLAUDE_GPT, ""),
    // Node SDKs.
    ("E1+B1+S1+S2",       "openai-node",   Runtime::Node,   "chat_basic",       GEN,      "R1"),
    ("E2+B1+S1",          "anthropic-ts",  Runtime::Node,   "messages_basic",   GEN,      "R1"),
    ("E3+TRN-1+CAT-9+B1", "openai-node",   Runtime::Node,   "responses_basic",  SESSION,  ""),
    ("E4",                "openai-node",   Runtime::Node,   "models_list",      ONE,      ""),
    ("E4",                "anthropic-ts",  Runtime::Node,   "models_list",      ONE,      ""),
    ("E1+T1+B1+S2",       "ai-sdk",        Runtime::Node,   "ai_sdk_openai",    GEN,      "R1"),
    ("E2+T1+B1",          "ai-sdk",        Runtime::Node,   "ai_sdk_anthropic", GEN,      "R1"),
    // The AI SDK's default OpenAI model is Responses (`openai(model)`, no `.chat`), most apps' call;
    // `@ai-sdk/openai-compatible` is the generic provider apps wire a gateway in with. T4 only
    // where the gateway serves a JSON schema: not Bedrock alone. OpenRouter's JSON schema on Claude
    // is CAT-6's (the catalog sweep, every Claude row's OpenRouter candidate).
    ("E3+TRN-1+T1+T4+B1", "ai-sdk",        Runtime::Node,   "ai_sdk_responses", &[CLAUDE, GPT, FAILOVER, XAI, TOGETHER], "R1"),
    ("E3+TRN-1+T1+B1",    "ai-sdk",        Runtime::Node,   "ai_sdk_responses", &[OPENROUTER, BEDROCK], ""),
    ("E3+B1",             "ai-sdk",        Runtime::Node,   "ai_sdk_conversation", GEN,   ""),
    ("E1+T1+B1+S2",       "ai-sdk",        Runtime::Node,   "ai_sdk_compatible", GEN,     "R1"),
    ("T2",                "anthropic-ts",  Runtime::Node,   "thinking_replay",  &[CLAUDE], ""),
    ("T3",                "ai-sdk",        Runtime::Node,   "ai_sdk_vision",    CLAUDE_GPT, ""),
    ("T4",                "openai-node",   Runtime::Node,   "structured_chat",  FAMILIES, ""),
    ("T4",                "ai-sdk",        Runtime::Node,   "ai_sdk_structured", FAMILIES, ""),
    ("T5",                "openai-node",   Runtime::Node,   "reasoning_effort", &[CLAUDE, SONNET, GPT, XAI], ""),
    ("T6",                "openai-node",   Runtime::Node,   "typed_error",      CLAUDE_GPT, ""),
    ("T6",                "anthropic-ts",  Runtime::Node,   "typed_error",      CLAUDE_GPT, ""),
    ("T6",                "ai-sdk",        Runtime::Node,   "ai_sdk_error",     CLAUDE_GPT, ""),
    ("K1",                "openai-node",   Runtime::Node,   "auto_cache",       &[CLAUDE], ""),
    ("K1",                "ai-sdk",        Runtime::Node,   "ai_sdk_cache",     &[CLAUDE], ""),
    ("T1+B1",             "openai-node",   Runtime::Node,   "tools_chat",       GEN,      "R1"),
    ("T1+B1",             "anthropic-ts",  Runtime::Node,   "tools_messages",   GEN,      "R1"),
    // Coding agents fixing a failing test in a fixture repo (W*). pi runs once per API mode; pi
    // and opencode take models only from config, generated from /v1/models. Codex's GPT row is a
    // Codex-native one (D74 kept it off the Chat-first GPT row).
    ("W1",                "claude-code",   Runtime::Harness, "claude-code",     SESSION,  ""),
    ("W2+TRN-24",         "codex",         Runtime::Harness, "codex",           CODEX_ROWS, ""),
    ("W3",                "opencode",      Runtime::Harness, "opencode",        SESSION,  ""),
    ("W4",                "pi",            Runtime::Harness, "pi:chat",         SESSION,  ""),
    ("W4",                "pi",            Runtime::Harness, "pi:messages",     SESSION,  ""),
    ("W4",                "pi",            Runtime::Harness, "pi:responses",    SESSION,  ""),
    // The same sessions for E7, in their own cells so a cost mismatch never reads as a failed task
    // (and a failed task never hides one): the harness's model list, and the session cost it
    // displays against the ledger. Claude Code only where its own price table knows the model
    // (not the GPT row); Codex displays no cost and can't list the catalog.
    ("E7",                "claude-code",   Runtime::Harness, "claude-code",     &[CLAUDE, OPENROUTER], ""),
    ("E7",                "opencode",      Runtime::Harness, "opencode",        SESSION,  ""),
    ("E7",                "pi",            Runtime::Harness, "pi:chat",         SESSION,  ""),
    ("E7",                "pi",            Runtime::Harness, "pi:messages",     SESSION,  ""),
    ("E7",                "pi",            Runtime::Harness, "pi:responses",    SESSION,  ""),
    // Raw HTTP (`raw`): httpx on the wire, no SDK, the way a customer's own HTTP code calls us.
    ("E4",                "raw",           Runtime::Python, "raw_models",       ONE,      ""),
    ("E5+B4",             "raw",           Runtime::Python, "raw_count_compact", CLAUDE_GPT, ""),
    ("R1",                "raw",           Runtime::Python, "raw_failover",     &[FAILOVER], ""),
    ("R3",                "raw",           Runtime::Python, "raw_steer",        &[POOLED], ""),
    ("R4+B3",             "raw",           Runtime::Python, "raw_session_pin",  &[POOLED], ""),
    ("R5",                "raw",           Runtime::Python, "raw_big_body",     &[CLAUDE, FAILOVER], "R1"),
    ("B2+BIL-20",         "raw",           Runtime::Python, "raw_stream_abort", CLAUDE_GPT, ""),
    ("A1",                "raw",           Runtime::Python, "byo_raw",          CLAUDE_GPT, ""),
    ("K1+B3",             "raw",           Runtime::Python, "raw_auto_cache",   &[CLAUDE], ""),
    ("M1+B1",             "raw",           Runtime::Python, "raw_embeddings",   &[EMBED], ""),
    ("SEC-7",             "raw",           Runtime::Python, "leak_scan",        CLAUDE_GPT, ""),
    ("REL-22",            "raw",           Runtime::Python, "h2_burst",         CLAUDE_GPT, ""),
    // The remaining SDK pairings.
    ("M1+B1",             "langchain",     Runtime::Python, "langchain_embeddings", &[EMBED], ""),
    ("M1+B1",             "openai-node",   Runtime::Node,   "embeddings",       &[EMBED], ""),
    ("M1+B1",             "ai-sdk",        Runtime::Node,   "ai_sdk_embeddings", &[EMBED], ""),
    ("T4",                "openai-agents", Runtime::Python, "agents_structured", CLAUDE_GPT, ""),
    // TRN-7: thinking + tools from clients that never echo thinking (Claude direct, and Claude
    // over OpenRouter, where D77 / D79 lived).
    ("TRN-7+T1",          "openai-py",     Runtime::Python, "thinking_no_echo", &[CLAUDE, OPENROUTER], ""),
    ("TRN-7+T1",          "anthropic-py",  Runtime::Python, "thinking_no_echo", &[OPENROUTER], ""),
    // Coding agents behind the recording proxy (`harness_long.py cell`): every call they make is
    // held to the ledger one by one. Codex's thinking cells run native (codex) and translated
    // (claude), so they carry E3 too.
    ("R1",                "codex",         Runtime::Recorded, "codex+task",     &[FAILOVER], ""),
    ("R1",                "opencode",      Runtime::Recorded, "opencode+task",  &[FAILOVER], ""),
    ("R1",                "pi",            Runtime::Recorded, "pi:messages+task", &[FAILOVER], ""),
    ("R5",                "claude-code",   Runtime::Recorded, "claude-code+big", &[CLAUDE, FAILOVER], "R1"),
    ("T2+E3",             "codex",         Runtime::Recorded, "codex+thinking", &[CODEX, CLAUDE], ""),
    ("T2+E2",             "claude-code",   Runtime::Recorded, "claude-code+thinking", &[CLAUDE], ""),
    ("T2",                "pi",            Runtime::Recorded, "pi:messages+thinking", &[CLAUDE], ""),
    ("R4+B3",             "claude-code",   Runtime::Recorded, "claude-code+pin", &[POOLED], ""),
    ("R4+B3",             "pi",            Runtime::Recorded, "pi:messages+pin", &[POOLED_SONNET], ""),
    ("E5+B4",             "claude-code",   Runtime::Recorded, "claude-code+context", &[CLAUDE], ""),
    ("B2",                "claude-code",   Runtime::Recorded, "claude-code+abort", &[CLAUDE], ""),
    ("A1",                "claude-code",   Runtime::Recorded, "claude-code+byo", &[CLAUDE], ""),
    // Chat Completions from the coding agents that speak it (E1), each call's usage as the harness
    // saw it on the wire held to its row (E1, B1); opencode's turns replay the history it
    // accumulated from its streamed answers (S2, on the rows whose Chat stream the gateway builds
    // or OpenRouter sends, not OpenAI's own). Each fixes the fixture through its own tool loop: a
    // served turn must feed a tool call's result back (T1), relayed on GPT and OpenRouter,
    // translated to Messages on Claude.
    ("E1+S2+T1+B1",       "opencode",      Runtime::Recorded, "opencode+task",  &[CLAUDE, OPENROUTER], ""),
    ("E1+T1+B1",          "opencode",      Runtime::Recorded, "opencode+task",  &[GPT],    ""),
    ("E1+T1+B1",          "pi",            Runtime::Recorded, "pi:chat+task",   SESSION,  ""),
    // The same tool loop and per-call usage from Claude Code (Messages: native on Claude, translated
    // to Chat on GPT) and Codex (Responses: native on its own row, translated to Messages on Claude).
    ("T1+B1",             "claude-code",   Runtime::Recorded, "claude-code+task", CLAUDE_GPT, ""),
    ("T1+B1",             "codex",         Runtime::Recorded, "codex+task",     &[CODEX, CLAUDE], ""),
    // T3: pi attaches an image on each of its three wires, native and translated.
    ("T3",                "pi",            Runtime::Recorded, "pi:chat+vision", CLAUDE_GPT, ""),
    ("T3",                "pi",            Runtime::Recorded, "pi:messages+vision", CLAUDE_GPT, ""),
    ("T3",                "pi",            Runtime::Recorded, "pi:responses+vision", CLAUDE_GPT, ""),
    // S1: Claude Code's stream timed straight from Anthropic and through the gateway.
    ("S1",                "claude-code",   Runtime::Recorded, "claude-code+stream", &[CLAUDE], ""),
];

/// `(client, probe, route, claims)`: claims a cell carries on one route only, where the behavior
/// is specific to that route (Claude Code's request shape reaching native OpenAI).
const ROUTE_CLAIMS: &[(&str, &str, &str, &str)] =
    &[("claude-code", "claude-code", "gpt", "TRN-10")];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `.env` at the repo root, then the process environment (which wins).
fn env_keys() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Ok(s) = std::fs::read_to_string(repo_root().join(".env")) {
        for line in s.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                out.insert(k.trim().to_owned(), v.trim().trim_matches('"').to_owned());
            }
        }
    }
    for (k, v) in std::env::vars() {
        out.insert(k, v);
    }
    // No Bedrock key in the environment: mint a short-term one from the AWS credential chain.
    if !out.contains_key("AWS_BEARER_TOKEN_BEDROCK")
        && let Ok(o) = Command::new(interpreter(Runtime::Python))
            .arg(repo_root().join("verify/clients/py/bedrock_token.py"))
            .output()
        && o.status.success()
    {
        let token = String::from_utf8_lossy(&o.stdout).trim().to_owned();
        if !token.is_empty() {
            out.insert("AWS_BEARER_TOKEN_BEDROCK".to_owned(), token);
        }
    }
    out
}

fn interpreter(rt: Runtime) -> PathBuf {
    match rt {
        Runtime::Python => repo_root().join("verify/clients/py/.venv/bin/python"),
        Runtime::Node => PathBuf::from("node"),
        Runtime::Harness | Runtime::Recorded => {
            repo_root().join("verify/clients/py/.venv/bin/python")
        }
    }
}

fn probe_script(rt: Runtime) -> PathBuf {
    match rt {
        Runtime::Python => repo_root().join("verify/clients/py/probe.py"),
        Runtime::Node => repo_root().join("verify/clients/node/probe.mjs"),
        Runtime::Harness => repo_root().join("verify/clients/harness.py"),
        Runtime::Recorded => repo_root().join("verify/clients/harness_long.py"),
    }
}

/// The runtime's pinned dependencies are installed (the venv / `npm ci`).
fn installed(rt: Runtime) -> bool {
    match rt {
        Runtime::Python => interpreter(rt).exists(),
        Runtime::Node => repo_root()
            .join("verify/clients/node/node_modules/openai")
            .exists(),
        Runtime::Harness | Runtime::Recorded => {
            interpreter(rt).exists()
                && repo_root()
                    .join("verify/clients/node/node_modules/.bin/pi")
                    .exists()
        }
    }
}

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

fn main() {
    common::started();
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") {
        let keys = env_keys();
        for &(claims, client, rt, probe, routes, failover_claims) in CELLS {
            for route in routes {
                let have_keys = route
                    .pools
                    .iter()
                    .all(|(_, var)| keys.get(*var).is_some_and(|v| !v.is_empty()));
                if !have_keys || !installed(rt) || !gateway_bin().exists() {
                    continue;
                }
                let mut claims = claims.to_owned();
                if route.name == "failover" && !failover_claims.is_empty() {
                    claims = format!("{claims}+{failover_claims}");
                }
                // E7 cells re-run a W* session for its cost; route claims stay on the W* cell.
                for (c, p, r, extra) in ROUTE_CLAIMS {
                    if (*c, *p, *r) == (client, probe, route.name) && claims != "E7" {
                        claims = format!("{claims}+{extra}");
                    }
                }
                let name = format!("{claims}::{client}::{}::{probe}", route.name);
                let checks = Checks {
                    task: claims.split('+').any(|c| c != "E7"),
                    e7: claims.split('+').any(|c| c == "E7"),
                    claims: claims.clone(),
                };
                let (route, keys) = (*route, keys.clone());
                trials.push(Trial::test(name, move || {
                    common::judge(|| run_cell(rt, client, probe, route, &keys, &checks))
                }));
            }
        }
    }
    // Live traffic: no reconciliation window may be open while it runs (see common::live_traffic).
    let _traffic = (!trials.is_empty() && !args.list).then(common::live_traffic);
    trials.push(Trial::test(
        "harness_cells_report_inside_the_nextest_budget",
        harness_cells_report_inside_the_nextest_budget,
    ));
    libtest_mimic::run(&args, trials).exit();
}

/// A harness cell reports before nextest's kill, never as a bare TIMEOUT
/// (`E7::pi::gpt::pi:responses`, 2026-10-01: OpenAI sent no head for pi's fifth request in 172s
/// and nextest killed the cell at 180s). The client is told its deadline; one that runs past it
/// is killed with its process group and the cell gets the reason; a client that exits leaves no
/// subprocess holding its stdout open. And the session it was stopped in is read: a provider
/// that had the last request whole (an estimated, cancelled row) and sent no head for a minute
/// is unavailable, while a short wait, a request it never had, or a head that came are not.
/// claim: E7
fn harness_cells_report_inside_the_nextest_budget() -> Result<(), Failed> {
    let mut problems = Vec::new();
    let sh = |script: &str| {
        let mut c = Command::new("sh");
        c.args(["-c", script]);
        c
    };
    match common::run_client_until(
        &mut sh("echo VERIFY $VERIFY_DEADLINE_SECS"),
        Some(Duration::from_secs(40)),
    ) {
        Ok((_, out)) if String::from_utf8_lossy(&out).trim() == "VERIFY 40" => {}
        other => problems.push(format!("deadline not handed to the client: {other:?}")),
    }
    let at = Instant::now();
    match common::run_client_until(&mut sh("sleep 60"), Some(Duration::from_secs(1))) {
        Err(e) if e.contains("killed") && at.elapsed() < Duration::from_secs(15) => {}
        other => problems.push(format!(
            "a client past its deadline was not killed in time ({:?}): {other:?}",
            at.elapsed()
        )),
    }
    let at = Instant::now();
    match common::run_client_until(&mut sh("sleep 60 & echo done"), None) {
        Ok((_, out))
            if String::from_utf8_lossy(&out).trim() == "done"
                && at.elapsed() < Duration::from_secs(15) => {}
        other => problems.push(format!(
            "a subprocess left behind held the client's stdout ({:?}): {other:?}",
            at.elapsed()
        )),
    }
    let row = |outcome: &str, estimated: bool, ms: u64| {
        serde_json::json!({"request_id": "r-5", "provider": "openai", "outcome": outcome,
            "usage_estimated": estimated, "latency_ms": ms})
    };
    let headless = r#"{"fields":{"message":"upstream request errored","request_id":"r-5","error":"Downstream ConnectionClosed context: Prematurely before response header is sent"}}"#;
    for (what, rows, log, want) in [
        (
            "stalled",
            vec![row("client_cancelled", true, 172_208)],
            headless,
            true,
        ),
        (
            "a short wait",
            vec![row("client_cancelled", true, 20_000)],
            headless,
            false,
        ),
        (
            "never delivered",
            vec![row("client_cancelled", false, 172_208)],
            headless,
            false,
        ),
        (
            "a head came",
            vec![row("client_cancelled", true, 172_208)],
            "",
            false,
        ),
        ("answered", vec![row("ok", false, 172_208)], headless, false),
        ("no rows", vec![], headless, false),
    ] {
        if common::stalled_on_provider(&rows, log).is_some() != want {
            problems.push(format!("{what}: stalled_on_provider should be {want}"));
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; ").into())
    }
}

/// What a cell asserts beyond the ledger, from the claims in its name.
struct Checks {
    /// Any claim but E7: the client's verdict must hold (for a harness, the fixture test passes).
    task: bool,
    /// E7: the harness's model list and displayed session cost (see [`e7_problems`]). An E7-only
    /// cell doesn't need the task to succeed: a session that failed still has a cost to compare.
    e7: bool,
    /// The cell's claims, `+`-joined, handed to the probe as `VERIFY_CLAIMS`: a probe shared by
    /// routes that can and can't serve a feature asserts it only where its cell claims it.
    claims: String,
}

/// Kills its process on drop, so a failing cell never leaks a gateway or nats-server.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_ready(metrics_port: u16, gw: &mut Child, log: &Path) -> Result<(), Failed> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = gw.try_wait() {
            return Err(format!("gateway exited {status}: {}", tail(log)).into());
        }
        if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", metrics_port)) {
            // `Connection: close` and a read timeout: the admin server keeps an idle connection
            // open for a minute, and reading to EOF without them waits that long.
            let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
            let _ = s.write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
            let mut body = String::new();
            let _ = s.read_to_string(&mut body);
            if body.lines().any(|l| l.starts_with("ai_allowance_ready 1")) {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(format!("gateway never became ready: {}", tail(log)).into())
}

fn tail(path: &Path) -> String {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    let start = s.len().saturating_sub(2000);
    s[start..].to_owned()
}

/// One cell under the stock SDKs' own retry policy. Every probe runs its SDK with retries off, so
/// each HTTP call it records is one ledger row; the cell retries instead, whole, on a fresh
/// gateway. When the client failed on an answer its SDK retries (a 503 overloaded, a 429), the
/// cell waits as the SDK would (Retry-After, else its backoff) and runs again, up to the SDK's two
/// retries. If the last answer is still a provider's own, relayed because the route has no other
/// provider to fail over to, the cell is INCONCLUSIVE: it proves nothing about the gateway either
/// way. A gateway-made answer stays a failure.
fn run_cell(
    rt: Runtime,
    client: &str,
    probe: &str,
    route: Route,
    keys: &BTreeMap<String, String>,
    checks: &Checks,
) -> Result<(), Failed> {
    let mut retry = 0;
    loop {
        let mut retryable = None;
        let at = std::time::Instant::now();
        let result = attempt_cell(rt, client, probe, route, keys, checks, &mut retryable);
        let took = at.elapsed();
        let (Err(failure), Some((e, providers))) = (&result, retryable) else {
            return result;
        };
        let status = e["status"].as_u64().unwrap_or(0) as u16;
        let who = e["provider"].as_str().unwrap_or("the gateway");
        let why = format!(
            "{who} answered {status} on attempt {} of {} (request {}){}",
            retry + 1,
            common::SDK_MAX_RETRIES + 1,
            e["request_id"].as_str().unwrap_or("?"),
            if providers {
                ", and the route has no other provider to fail over to"
            } else {
                ""
            },
        );
        match common::sdk_retry(status, retry, took, |h| error_header(&e, h)) {
            Some(wait) => {
                eprintln!("{why}; retrying in {wait:?}, as the SDK would");
                std::thread::sleep(wait);
                retry += 1;
            }
            None if providers => {
                return Err(common::inconclusive(&why, failure.message().unwrap_or("")));
            }
            None => return result,
        }
    }
}

/// A response header from a probe's `errors` entry.
fn error_header(e: &Value, name: &str) -> Option<String> {
    let key = match name {
        "retry-after" => "retry_after",
        "retry-after-ms" => "retry_after_ms",
        "x-should-retry" => "should_retry",
        _ => return None,
    };
    e[key].as_str().map(str::to_owned)
}

/// The error answer the client failed on, when its SDK would retry it, and whether it is the
/// provider's own answer that the gateway had to relay: `x-beyond-provider` names who sent it,
/// the gateway's row for it records the same upstream status, and the route holds one live
/// provider, so there was nothing to fail over to.
fn retryable_failure(verdict: &Value, rows: &[Value], route: Route) -> Option<(Value, bool)> {
    let e = verdict["errors"].as_array()?.last()?;
    let status = e["status"].as_u64()?;
    if !common::sdk_retryable(status as u16, |h| error_header(e, h)) {
        return None;
    }
    let providers = route.pools.len() - route.dead.len() == 1
        && e["provider"].as_str().is_some_and(|provider| {
            rows.iter().any(|r| {
                r["request_id"] == e["request_id"]
                    && r["provider"] == provider
                    && r["outcome"] == "upstream_error"
                    && r["upstream_status"] == status
            })
        });
    Some((e.clone(), providers))
}

fn attempt_cell(
    rt: Runtime,
    client: &str,
    probe: &str,
    route: Route,
    keys: &BTreeMap<String, String>,
    checks: &Checks,
    retryable: &mut Option<(Value, bool)>,
) -> Result<(), Failed> {
    let dir = std::env::temp_dir().join(format!(
        "verify-live-{}-{}",
        std::process::id(),
        free_port()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let nats_port = free_port();
    let _nats = Guard(
        Command::new("nats-server")
            .args([
                "-js",
                "-a",
                "127.0.0.1",
                "-p",
                &nats_port.to_string(),
                "-sd",
            ])
            .arg(dir.join("nats"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("nats-server: {e}"))?,
    );

    let (port, metrics_port) = (free_port(), free_port());
    let mut cfg = format!(
        "listen = \"127.0.0.1:{port}\"\nmetrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\nupstream_tls = true\n\n[pool_keys]\n"
    );
    for (provider, var) in route.pools {
        cfg.push_str(&format!("{provider} = [{:?}]\n", keys[*var]));
    }
    if !route.dead.is_empty() {
        cfg.push_str("\n[provider_authorities]\n");
        for p in route.dead {
            cfg.push_str(&format!("{p} = \"127.0.0.1:9\"\n"));
        }
    }
    cfg.push_str(&format!("\n[signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"));
    cfg.push_str(common::DEV_ID_SIGNING_TOML);
    let cfg_path = dir.join("gateway.toml");
    std::fs::write(&cfg_path, cfg).map_err(|e| e.to_string())?;

    let log_path = dir.join("gateway.log");
    let log = std::fs::File::create(&log_path).map_err(|e| e.to_string())?;
    let mut gw = Guard(
        Command::new(gateway_bin())
            .args(["run", "-c"])
            .arg(&cfg_path)
            .env("AI_LOG", "warn,ai.usage=info")
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .spawn()
            .map_err(|e| format!("gateway: {e}"))?,
    );
    // nats needs a moment before the gateway's first connect; readiness waits for the scan.
    wait_ready(metrics_port, &mut gw.0, &log_path)?;

    let mut cmd = Command::new(interpreter(rt));
    cmd.arg(probe_script(rt));
    if matches!(rt, Runtime::Recorded) {
        cmd.arg("cell");
    }
    cmd.arg(probe)
        .env("VERIFY_BASE", format!("http://127.0.0.1:{port}"))
        .env("VERIFY_KEY", DEV_TOKEN)
        .env("VERIFY_MODEL", route.model)
        .env("VERIFY_PROVIDER", route.pools[0].0)
        .env("VERIFY_CLIENT", client)
        .env("VERIFY_CLAIMS", &checks.claims)
        .env("VERIFY_GATEWAY_PID", gw.0.id().to_string());
    // A BYO probe sends the provider's own key through /{provider}/, the way a customer with
    // their own key would; a leak probe looks for the pool key in everything it was sent; an S1
    // stream cell calls the provider directly for its baseline. Only those probes see a real key.
    if probe.starts_with("byo")
        || probe.ends_with("+byo")
        || probe.starts_with("leak")
        || probe.ends_with("+stream")
    {
        cmd.env("VERIFY_BYO_KEY", &keys[route.pools[0].1]);
    }
    // Inside nextest's time budget: the probe is told its deadline, and killed just past it.
    let (_, out) = common::run_client(cmd.stderr(Stdio::inherit()))
        .map_err(|e| format!("probe: {e}\n--- gateway log ---\n{}", tail(&log_path)))?;
    let stdout = String::from_utf8_lossy(&out);
    let verdict: Value = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("VERIFY "))
        .and_then(|j| serde_json::from_str(j).ok())
        .ok_or_else(|| format!("probe printed no VERIFY line:\n{stdout}"))?;

    // A harness stopped at its deadline: what it was waiting on decides the verdict, not the cost
    // or the task it didn't get to finish. A session whose provider had its last request and sent
    // nothing back is INCONCLUSIVE; anything else fails, with what the harness printed.
    if let Some(secs) = verdict["detail"]["timed_out"].as_u64() {
        std::thread::sleep(Duration::from_millis(500));
        if route.pools.len() - route.dead.len() == 1 {
            let log = std::fs::read_to_string(&log_path).unwrap_or_default();
            common::note_if_stalled_on_provider(&usage_rows(&log_path), &log);
        }
        return Err(format!(
            "the harness was stopped at the test's time budget after {secs}s, still running: {}\n\
             --- gateway log ---\n{}",
            verdict["detail"],
            tail(&log_path)
        )
        .into());
    }

    // A probe whose oracle rests on something the provider documents as best effort (OpenAI's
    // prompt cache hits) says so in `detail.best_effort` when it missed: the client saw nothing
    // the gateway did wrong, and nothing that proves it right. Its calls still go to the ledger
    // witness; a ledger problem is a failure, a clean ledger INCONCLUSIVE.
    let best_effort = (checks.task && verdict["ok"] != true && verdict["calls"].is_array())
        .then(|| verdict["detail"]["best_effort"].as_str().map(str::to_owned))
        .flatten();
    // Witness 1: the client's own verdict.
    if checks.task && verdict["ok"] != true && best_effort.is_none() {
        let rows = usage_rows(&log_path);
        *retryable = retryable_failure(&verdict, &rows, route);
        // A coding agent (harness.py, or harness_long.py's recorded cells) has already retried
        // with its own defaults and reports no `errors`: a session that ended on its only live
        // provider's retryable answer is INCONCLUSIVE (`common::judge`), as LNG sessions are.
        if matches!(rt, Runtime::Harness | Runtime::Recorded)
            && route.pools.len() - route.dead.len() == 1
        {
            common::note_if_ended_unavailable(&rows);
        }
        return Err(format!(
            "client verdict failed: {}\n--- gateway log ---\n{}",
            verdict["detail"],
            tail(&log_path)
        )
        .into());
    }

    // Witness 2: the ledger. Rows can land a beat after the response; give them a moment.
    if verdict["recorded"] == true {
        let problems = recorded_problems(&verdict, &log_path, route);
        export_rows(&log_path);
        let _ = std::fs::remove_dir_all(&dir);
        return if problems.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "ledger disagrees with the recorded calls:\n  {}\n--- detail --- {}",
                problems.join("\n  "),
                verdict["detail"]
            )
            .into())
        };
    }
    // A harness's individual HTTP calls aren't visible to us (`calls` is null): check the ledger
    // in aggregate instead.
    if verdict["calls"].is_null() {
        std::thread::sleep(Duration::from_millis(500));
        let rows = usage_rows(&log_path);
        let mut problems = Vec::new();
        if rows.is_empty() {
            problems.push("the harness finished but the gateway billed nothing".to_owned());
        }
        for row in &rows {
            if row["provider"] != route.serves {
                problems.push(format!(
                    "row served by {}, route expects {} ({row})",
                    row["provider"], route.serves
                ));
            }
            if row["usage_estimated"] == true {
                problems.push(format!("row is an estimate ({row})"));
            }
            if row["price_model"].as_str().is_none_or(str::is_empty) {
                problems.push(format!("row's price_model doesn't resolve ({row})"));
            }
        }
        if rows
            .iter()
            .map(|r| r["output_tokens"].as_u64().unwrap_or(0))
            .sum::<u64>()
            == 0
        {
            problems.push("no output tokens billed across the session".to_owned());
        }
        if checks.e7 {
            problems.extend(e7_problems(&verdict["detail"], &rows, route.model));
        }
        export_rows(&log_path);
        let _ = std::fs::remove_dir_all(&dir);
        return if problems.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "ledger: {}\n--- detail --- {}",
                problems.join("; "),
                verdict["detail"]
            )
            .into())
        };
    }
    let calls = verdict["calls"].as_array().cloned().unwrap_or_default();
    // A call the probe expects no row for (a free token count, a BYO key) must not get one; every
    // other call gets exactly one. A cancelled call's row lands when the gateway notices the
    // client gone, so cells with one wait longer for it.
    let rowed = calls.iter().filter(|c| c["expect"]["rows"] != 0).count();
    let slow = calls.iter().any(|c| c["expect"]["estimated"] == true);
    let deadline = Instant::now() + Duration::from_secs(if slow { 30 } else { 5 });
    let rows = loop {
        let rows = usage_rows(&log_path);
        if rows.len() >= rowed || Instant::now() >= deadline {
            break rows;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    // A free call's row would land as fast as any other; give a stray one the same beat.
    if rowed < calls.len() {
        std::thread::sleep(Duration::from_millis(500));
    }
    let rows = if rowed < calls.len() {
        usage_rows(&log_path)
    } else {
        rows
    };
    let mut problems = Vec::new();
    if rows.len() != rowed {
        problems.push(format!(
            "{} client calls want a row, but there are {} ai.usage rows",
            rowed,
            rows.len()
        ));
    }
    for call in &calls {
        problems.extend(call_problems(call, &rows, route));
    }
    // BIL-7 / BIL-13 on every row, matched or not: the row names its wire and its price key.
    for row in &rows {
        if row["outcome"] == "ok"
            && !matches!(row["usage_wire"].as_str(), Some("openai" | "anthropic"))
        {
            problems.push(format!("row carries no usage_wire ({row})"));
        }
        if row["provider"].is_string() && row["price_model"].as_str().is_none_or(str::is_empty) {
            problems.push(format!("row's price_model doesn't resolve ({row})"));
        }
    }
    export_rows(&log_path);
    let _ = std::fs::remove_dir_all(&dir);
    if problems.is_empty() {
        match best_effort {
            None => Ok(()),
            Some(why) => {
                common::provider_unavailable(why);
                Err(format!("client verdict: {}", verdict["detail"]).into())
            }
        }
    } else {
        Err(format!(
            "ledger disagrees with the client:\n  {}\n--- detail --- {}",
            problems.join("\n  "),
            verdict["detail"]
        )
        .into())
    }
}

/// The paths the gateway bills (one `ai.usage` row per served call); any other path is free.
const BILLED_SUFFIXES: &[&str] = &[
    "/v1/messages",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/responses/compact",
    "/v1/embeddings",
];

/// A recorded harness session against the ledger, call by call. Every HTTP call the harness made
/// went through the recording proxy, so each one is held to its rows by `x-beyond-request-id`:
///
/// - a served billed call (POST to a [`BILLED_SUFFIXES`] path, 2xx): exactly one row, on the
///   route's provider, not an estimate, with output (unless the client hung up first);
/// - a refused billed call: at most one row, billing nothing;
/// - a free call (`/v1/models`, `count_tokens`): no row;
/// - `expect` on a call overrides that, with the keys a probe uses (`rows: 0`, `estimated` with
///   `output_min`, `provider`, `row_min`);
/// - and no row belongs to no call. With `same_provider` in the verdict, every row names one
///   provider (a session pin).
fn recorded_problems(verdict: &Value, log: &Path, route: Route) -> Vec<String> {
    let n = |v: &Value| v.as_u64().unwrap_or(0);
    let calls = verdict["calls"].as_array().cloned().unwrap_or_default();
    let is_billed = |c: &Value| {
        c["method"] == "POST"
            && c["path"]
                .as_str()
                .is_some_and(|p| BILLED_SUFFIXES.iter().any(|s| p.ends_with(s)))
    };
    let ok_status = |c: &Value| (200..300).contains(&n(&c["status"]));
    let wants_row = |c: &Value| {
        c["expect"]["estimated"] == true
            || (c["expect"]["rows"] != 0 && is_billed(c) && ok_status(c))
    };
    let rowed = calls.iter().filter(|c| wants_row(c)).count();
    let slow = calls.iter().any(|c| c["expect"]["estimated"] == true);
    let deadline = Instant::now() + Duration::from_secs(if slow { 30 } else { 5 });
    let mut rows = usage_rows(log);
    while rows.len() < rowed && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
        rows = usage_rows(log);
    }
    // A free or refused call's row would land as fast as any other; give a stray one a beat.
    std::thread::sleep(Duration::from_millis(500));
    let rows = usage_rows(log);
    let mut problems = Vec::new();
    let mut used = 0;
    for call in &calls {
        let label = format!(
            "{} {} -> {} ({})",
            call["method"].as_str().unwrap_or("?"),
            call["path"].as_str().unwrap_or("?"),
            call["status"],
            call["request_id"].as_str().unwrap_or("no request id")
        );
        let mine: Vec<&Value> = match call["request_id"].as_str() {
            Some(id) => rows.iter().filter(|r| r["request_id"] == id).collect(),
            None => Vec::new(),
        };
        used += mine.len();
        let expect = &call["expect"];
        if !wants_row(call) {
            // Free, BYO, or refused: no row, or (refused) one that bills nothing.
            let billed: u64 = mine
                .iter()
                .map(|r| {
                    n(&r["input_tokens"])
                        + n(&r["output_tokens"])
                        + n(&r["cache_read_tokens"])
                        + n(&r["cache_write_tokens"])
                })
                .sum();
            let refused = is_billed(call) && !ok_status(call) && expect["rows"] != 0;
            if (refused && (mine.len() > 1 || billed != 0)) || (!refused && !mine.is_empty()) {
                problems.push(format!("{label}: want no billed row, got {mine:?}"));
            }
            continue;
        }
        let [row] = mine.as_slice() else {
            problems.push(format!("{label}: {} ai.usage rows, want 1", mine.len()));
            continue;
        };
        let serves = expect["provider"].as_str().unwrap_or(route.serves);
        if !serves.is_empty() && row["provider"] != serves {
            problems.push(format!(
                "{label}: served by {}, want {serves}",
                row["provider"]
            ));
        }
        if row["price_model"].as_str().is_none_or(str::is_empty) {
            problems.push(format!("{label}: price_model doesn't resolve ({row})"));
        }
        if expect["estimated"] == true {
            if row["usage_estimated"] != true {
                problems.push(format!(
                    "{label}: a cut-short call's row is not flagged usage_estimated ({row})"
                ));
            }
            if n(&row["input_tokens"])
                + n(&row["cache_read_tokens"])
                + n(&row["cache_write_tokens"])
                == 0
            {
                problems.push(format!(
                    "{label}: a cut-short call's estimate bills no input ({row})"
                ));
            }
            if n(&row["output_tokens"]) < n(&expect["output_min"]) {
                problems.push(format!(
                    "{label}: estimate bills {} output tokens, want >= {} ({row})",
                    row["output_tokens"], expect["output_min"]
                ));
            }
            continue;
        }
        if row["usage_estimated"] == true {
            problems.push(format!("{label}: a completed call billed as an estimate"));
        }
        // The usage the harness was shown on the wire, where the cell's claims hold it to the
        // ledger (E1, E2, B3, B1).
        if let Some(u) = call["usage"].as_object() {
            problems.extend(usage_problems(&label, u, row));
        }
        if call["client_gone"] != true && n(&row["output_tokens"]) == 0 {
            problems.push(format!(
                "{label}: a completed call billed no output ({row})"
            ));
        }
        if let Some(mins) = expect["row_min"].as_object() {
            for (field, min) in mins {
                if n(&row[field.as_str()]) < n(min) {
                    problems.push(format!(
                        "{label}: row {field} = {}, want >= {min} ({row})",
                        row[field.as_str()]
                    ));
                }
            }
        }
    }
    if used != rows.len() {
        problems.push(format!(
            "{} ai.usage rows belong to no call the harness made",
            rows.len() - used
        ));
    }
    if verdict["same_provider"] == true {
        let providers: std::collections::BTreeSet<&str> =
            rows.iter().filter_map(|r| r["provider"].as_str()).collect();
        if providers.len() != 1 {
            problems.push(format!(
                "the session's rows name providers {providers:?}, want one (a session pin)"
            ));
        }
    }
    problems
}

/// One client call against the ledger. A probe can shape what the call must find with `expect`:
///
/// - `rows: 0` — no row may carry its request id (a free token count, a BYO key).
/// - `error: true` — the provider refused it: one row, zero tokens billed.
/// - `estimated: true` — the client cut it short: one row flagged `usage_estimated`, with input
///   (and `output_min`, default 0) tokens; `input_max` / `output_max` bound it by the same request
///   completed (BIL-20: an estimate never exceeds the truth).
/// - `provider` — the provider that must serve it, overriding the route's.
/// - `row_min` — `{field: n}`: the row's field is at least n (cache reads, server tool calls).
///
/// A call with no request id (cancelled before the response head) is matched by `tag`, the value
/// it sent as `x-beyond-metadata: {"verify": tag}`.
fn call_problems(call: &Value, rows: &[Value], route: Route) -> Vec<String> {
    let n = |v: &Value| v.as_u64().unwrap_or(0);
    let expect = &call["expect"];
    let mut problems = Vec::new();
    // A rowless call that carried no request id (a BYO call: the gateway only relays) has nothing
    // to match; the row count over the whole cell still proves it wrote none.
    if expect["rows"] == 0 && call["request_id"].is_null() {
        return problems;
    }
    let (label, matching): (String, Vec<&Value>) = if let Some(id) = call["request_id"].as_str() {
        (
            id.to_owned(),
            rows.iter().filter(|r| r["request_id"] == id).collect(),
        )
    } else if let Some(tag) = call["tag"].as_str() {
        let tagged = |r: &&Value| {
            r["metadata"]
                .as_str()
                .and_then(|m| serde_json::from_str::<Value>(m).ok())
                .is_some_and(|m| m["verify"] == tag)
        };
        (format!("tag {tag}"), rows.iter().filter(tagged).collect())
    } else {
        return vec![format!("a call carried no x-beyond-request-id: {call}")];
    };
    if expect["rows"] == 0 {
        if !matching.is_empty() {
            problems.push(format!(
                "{label}: billed {} row(s), want none ({:?})",
                matching.len(),
                matching
            ));
        }
        return problems;
    }
    let [row] = matching.as_slice() else {
        return vec![format!(
            "{label}: {} ai.usage rows, want exactly 1",
            matching.len()
        )];
    };
    let serves = expect["provider"].as_str().unwrap_or(route.serves);
    if !serves.is_empty() && row["provider"] != serves {
        problems.push(format!(
            "{label}: served by {}, want {serves}",
            row["provider"]
        ));
    }
    if let Some(mins) = expect["row_min"].as_object() {
        for (field, min) in mins {
            if n(&row[field.as_str()]) < n(min) {
                problems.push(format!(
                    "{label}: row {field} = {}, want >= {min} ({row})",
                    row[field.as_str()]
                ));
            }
        }
    }
    if expect["error"] == true {
        let billed: u64 = [
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ]
        .iter()
        .map(|k| n(&row[*k]))
        .sum();
        if billed != 0 || row["outcome"] == "ok" {
            problems.push(format!(
                "{label}: a refused call billed {billed} tokens, outcome {} ({row})",
                row["outcome"]
            ));
        }
        return problems;
    }
    if expect["estimated"] == true {
        if row["usage_estimated"] != true {
            problems.push(format!(
                "{label}: a cut-short call's row is not flagged usage_estimated ({row})"
            ));
        }
        let anthropic_wire = row["usage_wire"] == "anthropic";
        let input = n(&row["input_tokens"])
            + if anthropic_wire {
                n(&row["cache_read_tokens"]) + n(&row["cache_write_tokens"])
            } else {
                0
            };
        let output = n(&row["output_tokens"]);
        if input == 0 {
            problems.push(format!(
                "{label}: a cut-short call's estimate bills no input ({row})"
            ));
        }
        if output < n(&expect["output_min"]) {
            problems.push(format!(
                "{label}: estimate bills {output} output tokens, want >= {} ({row})",
                expect["output_min"]
            ));
        }
        for (k, got) in [("input_max", input), ("output_max", output)] {
            if let Some(max) = expect[k].as_u64()
                && got > max
            {
                problems.push(format!(
                    "{label}: estimate {got} exceeds the completed request's {max} ({k}; {row})"
                ));
            }
        }
        return problems;
    }
    let Some(u) = call["usage"].as_object() else {
        // A client that opted out of usage (stream_options.include_usage false) is still billed
        // exactly: the row can't be an estimate, and output was generated.
        if expect["exact"] == true {
            if row["usage_estimated"] == true || n(&row["output_tokens"]) == 0 {
                problems.push(format!(
                    "{label}: a call the client asked no usage for isn't billed exactly ({row})"
                ));
            }
        } else {
            problems.push(format!("{label}: the client was shown no usage"));
        }
        return problems;
    };
    problems.extend(usage_problems(&label, u, row));
    if row["usage_estimated"] == true {
        problems.push(format!(
            "{label}: row is an estimate on a completed call ({row})"
        ));
    }
    problems
}

/// The usage a client was shown (`input_total`, `output`, and the optional breakouts) against its
/// row, normalized for the row's wire: tokens, cache reads (B3 / BIL-8), reasoning (BIL-9) and
/// the service tier (BIL-11).
fn usage_problems(label: &str, u: &serde_json::Map<String, Value>, row: &Value) -> Vec<String> {
    let n = |v: &Value| v.as_u64().unwrap_or(0);
    let mut problems = Vec::new();
    let anthropic_wire = match row["usage_wire"].as_str() {
        Some(w) => w == "anthropic",
        None => matches!(row["provider"].as_str(), Some("anthropic" | "bedrock")),
    };
    let row_input = n(&row["input_tokens"])
        + if anthropic_wire {
            n(&row["cache_read_tokens"]) + n(&row["cache_write_tokens"])
        } else {
            0
        };
    if row_input != n(&u["input_total"]) {
        problems.push(format!(
            "{label}: client saw {} input tokens, row bills {row_input} ({row})",
            u["input_total"]
        ));
    }
    // `output_with_reasoning`: a client that reports reasoning apart from output without saying
    // which convention the provider used (the AI SDK) — either exact count is right.
    let alt = u.get("output_with_reasoning").map(n);
    if n(&row["output_tokens"]) != n(&u["output"]) && alt != Some(n(&row["output_tokens"])) {
        problems.push(format!(
            "{label}: client saw {} output tokens, row bills {} ({row})",
            u["output"], row["output_tokens"]
        ));
    }
    // B3 / BIL-8: cache reads are the same number on both sides.
    if let Some(cr) = u.get("cache_read")
        && n(cr) != n(&row["cache_read_tokens"])
    {
        problems.push(format!(
            "{label}: client saw {cr} cache-read tokens, row bills {} ({row})",
            row["cache_read_tokens"]
        ));
    }
    // BIL-9: reasoning is counted once — the row's breakout equals what the client was shown.
    if let Some(r) = u.get("reasoning").filter(|r| n(r) > 0)
        && reasoning_of(row) != Some(n(r))
    {
        problems.push(format!(
            "{label}: client saw {r} reasoning tokens, row reports {} ({row})",
            row["reasoning_tokens"]
        ));
    }
    // BIL-11: the service tier the provider echoed is the one recorded.
    if let Some(tier) = u.get("service_tier").filter(|t| t.is_string())
        && &row["service_tier"] != tier
    {
        problems.push(format!(
            "{label}: client was shown service_tier {tier}, row records {} ({row})",
            row["service_tier"]
        ));
    }
    problems
}

/// The row's `reasoning_tokens`, logged as `Debug` of an `Option` (`Some(12)` / `None`) so a real
/// zero stays distinct from "not reported".
fn reasoning_of(row: &Value) -> Option<u64> {
    match &row["reasoning_tokens"] {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.strip_prefix("Some(")?.strip_suffix(')')?.parse().ok(),
        _ => None,
    }
}

/// How far a harness's displayed session cost may sit from the ledger priced at the card: its own
/// float rounding, never a missing call or a mispriced token class.
const E7_TOLERANCE: f64 = 0.02;

/// Claim E7 on a harness cell: the harness listed the catalog (`detail.listing`), and the session
/// cost it displayed (`detail.cost.displayed`) equals the ledger rows priced at the model's card
/// (`detail.cost.pricing`, USD per million tokens), with opencode's cache writes at the input
/// rate, as opencode prices them on any OpenAI-compatible provider (D157). A harness that
/// displayed no cost fails: E7 is
/// claimed only on cells where one is shown.
fn e7_problems(detail: &Value, rows: &[Value], model: &str) -> Vec<String> {
    let mut problems = Vec::new();
    if detail["listing"]["ok"] != true {
        problems.push(format!(
            "E7: the harness's model list differs from /v1/models: {}",
            detail["listing"]
        ));
    }
    let cost = &detail["cost"];
    let Some(displayed) = cost["displayed"].as_f64() else {
        problems.push(format!(
            "E7: the harness displayed no cost: {}",
            cost["why"]
        ));
        return problems;
    };
    let price = |k: &str| cost["pricing"][k].as_f64().unwrap_or(f64::NAN) / 1e6;
    // opencode reaches the gateway through `@ai-sdk/openai-compatible` (2.0.41, bundled in
    // opencode-ai 1.18.34), whose usage converter reads only `prompt_tokens` and
    // `prompt_tokens_details.cached_tokens` and sets `cacheWrite: undefined`: the gateway's (and
    // OpenRouter's) `prompt_tokens_details.cache_write_tokens` is never read. opencode's step cost
    // then takes `input = inputTokens - cacheRead - cacheWrite` with cacheWrite 0 (its fallbacks
    // read only anthropic / vertex / bedrock / venice provider metadata), so every token it asked
    // to have cache-written is priced at the input rate. The same happens calling OpenRouter
    // directly with that provider: a client limitation, not a gateway gap (E7's oracle). The
    // ledger still bills writes at the card's cache_write rate; only the harness's arithmetic is
    // reproduced here.
    let write_rate = if detail["harness"] == "opencode" {
        price("input")
    } else {
        price("cache_write")
    };
    let mut billed = 0.0;
    for row in rows {
        if row["requested_model"] != model {
            problems.push(format!(
                "E7: a row for {} can't be priced at {model}'s card ({row})",
                row["requested_model"]
            ));
            continue;
        }
        let n = |k: &str| row[k].as_u64().unwrap_or(0) as f64;
        let (input, read, write) = (
            n("input_tokens"),
            n("cache_read_tokens"),
            n("cache_write_tokens"),
        );
        // A row keeps its upstream wire's token semantics: Anthropic's input excludes the cache,
        // OpenAI's (and OpenRouter's) includes reads and writes. Rows don't say which wire
        // (`usage_wire`, when present, does); today the provider decides it — OpenRouter is
        // always reached on Chat Completions.
        let anthropic = match row["usage_wire"].as_str() {
            Some(wire) => wire == "anthropic",
            None => matches!(row["provider"].as_str(), Some("anthropic" | "bedrock")),
        };
        let fresh = if anthropic {
            input
        } else {
            input - read - write
        };
        billed += fresh * price("input")
            + n("output_tokens") * price("output")
            + read * price("cache_read")
            + write * write_rate;
    }
    // Written so a NaN (a price missing from the detail) fails rather than passes.
    let within = billed > 0.0 && (displayed - billed).abs() <= billed * E7_TOLERANCE;
    if !within {
        problems.push(format!(
            "E7: the harness displayed ${displayed:.6}, the ledger priced at the card is \
             ${billed:.6} ({:+.1}%); cost {cost}",
            (displayed / billed - 1.0) * 100.0
        ));
    }
    problems
}

/// With `VERIFY_ROWS_OUT=<file>`, append the cell's billing rows there as JSON lines, so a run's
/// spend can be totalled afterwards.
fn export_rows(log: &Path) {
    let Some(out) = std::env::var_os("VERIFY_ROWS_OUT") else {
        return;
    };
    let lines: String = usage_rows(log).iter().map(|r| format!("{r}\n")).collect();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
    {
        let _ = f.write_all(lines.as_bytes());
    }
}

fn usage_rows(log: &Path) -> Vec<Value> {
    let Ok(f) = std::fs::File::open(log) else {
        return Vec::new();
    };
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| l.contains("\"ai.usage\""))
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .map(|v| v.get("fields").cloned().unwrap_or(v))
        .collect()
}
