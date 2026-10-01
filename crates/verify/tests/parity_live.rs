//! Differential parity: the gateway's core promise is "same as calling the provider".
//!
//! A seeded corpus (deterministic per `PARITY_SEED`, default 1) of ~36 cheap logical requests —
//! plain text, system prompts, multi-turn, unicode, stop sequences, max-token cut-offs, tools under
//! every `tool_choice`, parallel calls, tool results fed back, `json_schema` output, a tiny
//! base64 PNG carrying a code word, reasoning effort, and two invalid requests, each streamed and
//! not — is rendered into each dialect and sent twice per path: once **directly** to the provider
//! with the real key, once **through the gateway** with a managed key, same model. The two answers
//! are compared structurally, never textually (models are nondeterministic):
//!
//! * Same-dialect paths (a byte relay): HTTP status class, error type and code, the response's
//!   type skeleton (keys, value types, content-block / item / object discriminants), the raw finish
//!   reason, tool-call names, argument validity against the tool's schema, `json_schema` validity,
//!   whether the code word came back, usage presence; for streams the event-type sequence modulo
//!   repetition, each event type's key set, the terminal event, and that events arrive spread out
//!   (S1) rather than in one burst.
//! * Cross-dialect paths (the translation promise): a client in dialect X through the gateway to a
//!   model whose native dialect is Y, against the same logical request sent natively and directly:
//!   status class, the client's own error envelope, finish class, tool names, argument and
//!   structured-output validity, the code word, usage presence and comparable input size.
//!
//! Every gateway answer is also checked against the ledger: exactly one `ai.usage` row with its
//! `x-beyond-request-id`, served by the expected provider, whose tokens equal what the client was
//! shown (normalized for the wire).
//!
//! Allowed differences (crates/gateway/ARCHITECTURE.md): ids, timestamps and other values (only
//! types are compared), `x-beyond-*` headers (headers aren't compared), and the usage chunk the
//! gateway injects into an OpenAI-wire Chat stream whose client didn't ask for one ("inject
//! stream_options.include_usage").
//!
//! A mismatch is retried once (both sides) before it fails, so a single nondeterministic answer
//! doesn't read as a defect; the trial's stderr says when the first attempt disagreed.
//!
//! Trials are named `CLAIMS::raw::ROUTE::parity_CASE` and listed only with `VERIFY_LIVE=1`, the
//! gateway binary built, and every key the path needs; a missing key means the trial isn't listed.
//! A run costs about $0.30 (the estimate is printed first); raw answers of failing trials and a cost
//! line per trial land under `target/verify-parity/`.
//! `PARITY_DUMP=1` (with `--nocapture`) prints both sides of every comparison.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::{Value, json};

/// The dev signing key (seed `[7; 32]`, kid 1) and the tenant-1 token minted from it; the same
/// constants `mise run ai:mint-dev-key` prints (and `live.rs` uses).
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

// --- Providers, models and paths -----------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dialect {
    Chat,
    Responses,
    Messages,
}

impl Dialect {
    fn gateway_path(self) -> &'static str {
        match self {
            Dialect::Chat => "/v1/chat/completions",
            Dialect::Responses => "/v1/responses",
            Dialect::Messages => "/v1/messages",
        }
    }
}

/// A provider reached directly: its key, its base URL, and the gateway's pool-key name for it.
#[derive(Clone, Copy)]
struct Provider {
    /// The `[pool_keys]` entry, and the `provider` every ledger row must name.
    name: &'static str,
    var: &'static str,
    base: &'static str,
}

const OPENAI: Provider = Provider {
    name: "openai",
    var: "OPENAI_API_KEY",
    base: "https://api.openai.com/v1",
};
const ANTHROPIC: Provider = Provider {
    name: "anthropic",
    var: "ANTHROPIC_API_KEY",
    base: "https://api.anthropic.com/v1",
};
const XAI: Provider = Provider {
    name: "xai",
    var: "XAI_API_KEY",
    base: "https://api.x.ai/v1",
};
const OPENROUTER: Provider = Provider {
    name: "openrouter",
    var: "OPENROUTER_API_KEY",
    base: "https://openrouter.ai/api/v1",
};

/// A catalog model and how it's reached natively.
#[derive(Clone, Copy)]
struct Model {
    /// The catalog name the gateway is asked for.
    gw: &'static str,
    /// The provider's own id.
    direct: &'static str,
    provider: Provider,
    /// The dialect a direct call uses for a cross-dialect comparison (the row's primary).
    native: Dialect,
    /// An OpenAI reasoning model: no sampling, `max_completion_tokens`, effort `minimal` by default.
    openai_reasoning: bool,
    /// Reasons whether asked or not (grok): give it room before its answer.
    always_reasons: bool,
    /// Accepts `temperature: 0`.
    sampling: bool,
    /// USD per million input and output tokens (the catalog card).
    price: (f64, f64),
}

const GPT4O_MINI: Model = Model {
    gw: "gpt-4o-mini",
    direct: "gpt-4o-mini",
    provider: OPENAI,
    native: Dialect::Chat,
    openai_reasoning: false,
    always_reasons: false,
    sampling: true,
    price: (0.15, 0.6),
};
const GPT5_MINI: Model = Model {
    gw: "gpt-5-mini",
    direct: "gpt-5-mini",
    provider: OPENAI,
    native: Dialect::Chat,
    openai_reasoning: true,
    always_reasons: false,
    sampling: false,
    price: (0.25, 2.0),
};
const HAIKU: Model = Model {
    gw: "claude-haiku-4-5",
    direct: "claude-haiku-4-5",
    provider: ANTHROPIC,
    native: Dialect::Messages,
    openai_reasoning: false,
    always_reasons: false,
    sampling: true,
    price: (1.0, 5.0),
};
const GROK: Model = Model {
    gw: "grok-4.3",
    direct: "grok-4.3",
    provider: XAI,
    native: Dialect::Chat,
    openai_reasoning: false,
    always_reasons: true,
    sampling: true,
    price: (1.25, 2.5),
};
const SONNET4_OR: Model = Model {
    gw: "claude-sonnet-4",
    direct: "anthropic/claude-sonnet-4",
    provider: OPENROUTER,
    native: Dialect::Chat,
    openai_reasoning: false,
    always_reasons: false,
    sampling: true,
    price: (3.0, 15.0),
};

/// One comparison path: a client dialect, a model, and the claims every trial on it proves.
#[derive(Clone, Copy)]
struct ParityPath {
    route: &'static str,
    client: Dialect,
    model: Model,
    claims: &'static [&'static str],
}

impl ParityPath {
    fn cross(&self) -> bool {
        self.client != self.direct_dialect()
    }
    /// A same-dialect path calls the provider in the client's dialect; a cross-dialect one in the
    /// model's native dialect.
    fn direct_dialect(&self) -> Dialect {
        match (self.client, self.model.provider.name) {
            (Dialect::Responses, "openai") => Dialect::Responses,
            _ if self.client == self.model.native => self.client,
            _ => self.model.native,
        }
    }
}

const PATHS: &[ParityPath] = &[
    // Same dialect: the gateway relays.
    ParityPath {
        route: "openai-chat",
        client: Dialect::Chat,
        model: GPT4O_MINI,
        claims: &["E1"],
    },
    ParityPath {
        route: "openai-responses",
        client: Dialect::Responses,
        model: GPT5_MINI,
        claims: &["E3"],
    },
    ParityPath {
        route: "anthropic-messages",
        client: Dialect::Messages,
        model: HAIKU,
        claims: &["E2"],
    },
    ParityPath {
        route: "xai-chat",
        client: Dialect::Chat,
        model: GROK,
        claims: &["E1"],
    },
    ParityPath {
        route: "openrouter-chat",
        client: Dialect::Chat,
        model: SONNET4_OR,
        claims: &["E1"],
    },
    // Cross dialect: the gateway translates.
    ParityPath {
        route: "chat-to-claude",
        client: Dialect::Chat,
        model: HAIKU,
        claims: &["E1"],
    },
    ParityPath {
        route: "responses-to-claude",
        client: Dialect::Responses,
        model: HAIKU,
        claims: &["E3"],
    },
    ParityPath {
        route: "messages-to-gpt",
        client: Dialect::Messages,
        model: GPT5_MINI,
        claims: &["E2"],
    },
];

// --- The corpus ----------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Text,
    System,
    Multiturn,
    Unicode,
    Stop,
    Length,
    ToolsAuto,
    ToolsForced,
    ToolsRequired,
    ToolsNone,
    ParallelTools,
    ToolResult,
    ParallelToolResults,
    JsonSchema,
    Image,
    Reasoning,
    ErrorRole,
    ErrorToolChoice,
}

const KINDS: &[(Kind, &str)] = &[
    (Kind::Text, "text"),
    (Kind::System, "system"),
    (Kind::Multiturn, "multiturn"),
    (Kind::Unicode, "unicode"),
    (Kind::Stop, "stop"),
    (Kind::Length, "length"),
    (Kind::ToolsAuto, "tools_auto"),
    (Kind::ToolsForced, "tools_forced"),
    (Kind::ToolsRequired, "tools_required"),
    (Kind::ToolsNone, "tools_none"),
    (Kind::ParallelTools, "parallel_tools"),
    (Kind::ToolResult, "tool_result"),
    (Kind::ParallelToolResults, "parallel_tool_results"),
    (Kind::JsonSchema, "json_schema"),
    (Kind::Image, "image"),
    (Kind::Reasoning, "reasoning"),
    (Kind::ErrorRole, "error_role"),
    (Kind::ErrorToolChoice, "error_tool_choice"),
];

impl Kind {
    fn is_error(self) -> bool {
        matches!(self, Kind::ErrorRole | Kind::ErrorToolChoice)
    }
    fn is_tools(self) -> bool {
        matches!(
            self,
            Kind::ToolsAuto
                | Kind::ToolsForced
                | Kind::ToolsRequired
                | Kind::ToolsNone
                | Kind::ParallelTools
                | Kind::ToolResult
                | Kind::ParallelToolResults
        )
    }
}

#[derive(Clone)]
enum Part {
    Text(String),
    Png(String),
}

#[derive(Clone)]
enum Msg {
    System(String),
    User(Vec<Part>),
    Assistant(String),
    /// `(id, name, arguments)`.
    Calls(Vec<(String, String, Value)>),
    /// `(id, content)`.
    Results(Vec<(String, String)>),
    /// A message with a role no dialect has.
    BadRole(String),
}

#[derive(Clone, PartialEq)]
enum Choice {
    Unset,
    Auto,
    None,
    Required,
    Named(&'static str),
}

/// One logical request, dialect-free.
#[derive(Clone)]
struct Case {
    name: String,
    kind: Kind,
    stream: bool,
    msgs: Vec<Msg>,
    tools: bool,
    choice: Choice,
    schema: Option<Value>,
    stop: Option<Vec<String>>,
    /// `Some(n)`: the request's own token limit (the length case); else per model.
    max_tokens: Option<u32>,
    reasoning: bool,
    /// Ask for a usage chunk on a Chat stream (`stream_options.include_usage`).
    include_usage: bool,
    /// Every one must appear in the answer's text (case-insensitive) — compared as a boolean.
    expect: Vec<String>,
    /// None may appear in the answer's text.
    absent: Vec<String>,
}

/// splitmix64: a corpus is a pure function of its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[(self.next() % xs.len() as u64) as usize]
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

const WORDS: &[&str] = &[
    "MANGO", "ROBOT", "TIGER", "PIANO", "CACTUS", "VELVET", "HARBOR", "LEMON", "QUARTZ", "WIZARD",
    "FALCON", "JUNGLE", "COMET", "BISON", "MAPLE", "ORBIT",
];
const CITIES: &[&str] = &[
    "Paris", "Tokyo", "Lima", "Oslo", "Cairo", "Denver", "Hanoi", "Perth", "Quito", "Dublin",
];
const ZONES: &[&str] = &[
    "Europe/Berlin",
    "Asia/Tokyo",
    "America/Chicago",
    "Australia/Sydney",
];

fn tool_schema(name: &str) -> Value {
    match name {
        "get_weather" => json!({
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
            "additionalProperties": false,
        }),
        _ => json!({
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "IANA timezone"}},
            "required": ["timezone"],
            "additionalProperties": false,
        }),
    }
}

const TOOLS: &[(&str, &str)] = &[
    ("get_weather", "Get the current weather for a city."),
    (
        "get_time",
        "Get the current local time in an IANA timezone.",
    ),
];

fn answer_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "word": {"type": "string"},
            "count": {"type": "integer"},
            "colors": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["word", "count", "colors"],
        "additionalProperties": false,
    })
}

fn corpus(seed: u64) -> Vec<Case> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    for &(kind, kname) in KINDS {
        for stream in [false, true] {
            let w = rng.pick(WORDS).to_owned();
            let w2 = loop {
                let x = rng.pick(WORDS);
                if x != w {
                    break x.to_owned();
                }
            };
            let (ca, cb) = loop {
                let (a, b) = (rng.pick(CITIES), rng.pick(CITIES));
                if a != b {
                    break (a, b);
                }
            };
            let zone = rng.pick(ZONES);
            let n = rng.range(11, 39);
            let user = |s: String| Msg::User(vec![Part::Text(s)]);
            let mut c = Case {
                name: format!("{kname}{}", if stream { "_stream" } else { "" }),
                kind,
                stream,
                msgs: Vec::new(),
                tools: false,
                choice: Choice::Unset,
                schema: None,
                stop: None,
                max_tokens: None,
                reasoning: false,
                include_usage: stream && kind != Kind::Text,
                expect: Vec::new(),
                absent: Vec::new(),
            };
            match kind {
                Kind::Text => {
                    c.msgs = vec![user(format!(
                        "Reply with exactly the single word {w} and nothing else."
                    ))];
                    c.expect = vec![w.clone()];
                }
                Kind::System => {
                    c.msgs = vec![
                        Msg::System(format!(
                            "You are a relay. Whatever the user asks, reply with only the \
                             codeword {w} in uppercase and nothing else."
                        )),
                        user("What is the codeword?".to_owned()),
                    ];
                    c.expect = vec![w.clone()];
                }
                Kind::Multiturn => {
                    c.msgs = vec![
                        user(format!("Remember the codeword {w}.")),
                        Msg::Assistant("Understood, I will remember it.".to_owned()),
                        user("What was the codeword? Reply with just the word.".to_owned()),
                    ];
                    c.expect = vec![w.clone()];
                }
                Kind::Unicode => {
                    c.msgs = vec![user(format!(
                        "Repeat exactly this line and nothing else: {w} — café ✓ 日本"
                    ))];
                    c.expect = vec![w.clone(), "café".to_owned(), "日本".to_owned()];
                }
                Kind::Stop => {
                    c.msgs = vec![user(
                        "Count from 1 to 10 as digits separated by commas, nothing else."
                            .to_owned(),
                    )];
                    c.stop = Some(vec!["5".to_owned()]);
                    c.absent = vec!["6".to_owned()];
                }
                Kind::Length => {
                    c.msgs = vec![user(format!(
                        "Write a 500-word essay about the history of bridges in {ca}."
                    ))];
                    c.max_tokens = Some(16);
                }
                Kind::ToolsAuto => {
                    c.msgs = vec![user(format!(
                        "What's the weather in {ca} right now? Use the tool."
                    ))];
                    c.tools = true;
                }
                Kind::ToolsForced => {
                    c.msgs = vec![user(format!("I'm travelling to {ca} tomorrow."))];
                    c.tools = true;
                    c.choice = Choice::Named("get_weather");
                }
                Kind::ToolsRequired => {
                    c.msgs = vec![user(format!("What time is it in {zone}?"))];
                    c.tools = true;
                    c.choice = Choice::Required;
                }
                Kind::ToolsNone => {
                    c.msgs = vec![user(format!(
                        "Without using any tool, reply with only the word {w}."
                    ))];
                    c.tools = true;
                    c.choice = Choice::None;
                    c.expect = vec![w.clone()];
                }
                Kind::ParallelTools => {
                    c.msgs = vec![user(format!(
                        "Get the current weather in {ca} and in {cb}. Call the tool once per \
                         city, both calls at the same time."
                    ))];
                    c.tools = true;
                    c.choice = Choice::Auto;
                }
                Kind::ToolResult => {
                    let id = format!("call_{n}a");
                    c.msgs = vec![
                        user(format!(
                            "What's the weather in {ca}? Then tell me the code."
                        )),
                        Msg::Calls(vec![(
                            id.clone(),
                            "get_weather".to_owned(),
                            json!({"city": ca}),
                        )]),
                        Msg::Results(vec![(
                            id,
                            format!("{{\"temp_c\": {n}, \"code\": \"{w}\"}}"),
                        )]),
                    ];
                    c.tools = true;
                    c.expect = vec![w.clone()];
                }
                Kind::ParallelToolResults => {
                    let (ia, ib) = (format!("call_{n}a"), format!("call_{n}b"));
                    c.msgs = vec![
                        user(format!(
                            "What's the weather in {ca} and {cb}? Then tell me both codes."
                        )),
                        Msg::Calls(vec![
                            (ia.clone(), "get_weather".to_owned(), json!({"city": ca})),
                            (ib.clone(), "get_weather".to_owned(), json!({"city": cb})),
                        ]),
                        Msg::Results(vec![
                            (ia, format!("{{\"temp_c\": {n}, \"code\": \"{w}\"}}")),
                            (ib, format!("{{\"temp_c\": {}, \"code\": \"{w2}\"}}", n + 3)),
                        ]),
                    ];
                    c.tools = true;
                    c.expect = vec![w.clone(), w2.clone()];
                }
                Kind::JsonSchema => {
                    c.msgs = vec![user(format!(
                        "Return the word {w}, the number {n}, and two colors."
                    ))];
                    c.schema = Some(answer_schema());
                    c.expect = vec![w.clone()];
                }
                Kind::Image => {
                    c.msgs = vec![Msg::User(vec![
                        Part::Text(
                            "What single word is written in this image? Reply with just the word."
                                .to_owned(),
                        ),
                        Part::Png(png_base64(&w)),
                    ])];
                    c.expect = vec![w.clone()];
                }
                Kind::Reasoning => {
                    let (a, b) = (rng.range(12, 29), rng.range(12, 29));
                    c.msgs = vec![user(format!(
                        "What is {a} * {b}? Reply with just the number."
                    ))];
                    c.reasoning = true;
                    c.expect = vec![(a * b).to_string()];
                }
                Kind::ErrorRole => {
                    c.msgs = vec![
                        user("Hello.".to_owned()),
                        Msg::BadRole("I am not a real role.".to_owned()),
                    ];
                }
                Kind::ErrorToolChoice => {
                    c.msgs = vec![user(format!("What's the weather in {ca}?"))];
                    c.tools = true;
                    c.choice = Choice::Named("get_stock_price");
                }
            }
            out.push(c);
        }
    }
    out
}

// --- Rendering a case into a dialect -------------------------------------------------------------

fn max_tokens(case: &Case, m: &Model) -> u32 {
    if let Some(n) = case.max_tokens {
        return n;
    }
    if case.reasoning {
        return 2048;
    }
    if m.always_reasons {
        2000
    } else if m.openai_reasoning {
        1200
    } else {
        300
    }
}

fn render(case: &Case, d: Dialect, m: &Model, model_id: &str) -> Value {
    let mt = max_tokens(case, m);
    let mut b = serde_json::Map::new();
    b.insert("model".into(), json!(model_id));
    let temp = m.sampling && !case.reasoning;
    match d {
        Dialect::Chat => {
            let mut msgs = Vec::new();
            for msg in &case.msgs {
                match msg {
                    Msg::System(s) => msgs.push(json!({"role": "system", "content": s})),
                    Msg::User(parts) => {
                        msgs.push(json!({"role": "user", "content": chat_parts(parts)}))
                    }
                    Msg::Assistant(s) => msgs.push(json!({"role": "assistant", "content": s})),
                    Msg::Calls(calls) => msgs.push(json!({
                        "role": "assistant", "content": null,
                        "tool_calls": calls.iter().map(|(id, name, args)| json!({
                            "id": id, "type": "function",
                            "function": {"name": name, "arguments": args.to_string()},
                        })).collect::<Vec<_>>(),
                    })),
                    Msg::Results(rs) => {
                        for (id, content) in rs {
                            msgs.push(
                                json!({"role": "tool", "tool_call_id": id, "content": content}),
                            );
                        }
                    }
                    Msg::BadRole(s) => msgs.push(json!({"role": "robot", "content": s})),
                }
            }
            b.insert("messages".into(), json!(msgs));
            if case.tools {
                b.insert(
                    "tools".into(),
                    json!(TOOLS.iter().map(|(n, desc)| json!({
                        "type": "function",
                        "function": {"name": n, "description": desc, "parameters": tool_schema(n)},
                    })).collect::<Vec<_>>()),
                );
            }
            match &case.choice {
                Choice::Unset => {}
                Choice::Auto => _ = b.insert("tool_choice".into(), json!("auto")),
                Choice::None => _ = b.insert("tool_choice".into(), json!("none")),
                Choice::Required => _ = b.insert("tool_choice".into(), json!("required")),
                Choice::Named(n) => {
                    _ = b.insert(
                        "tool_choice".into(),
                        json!({"type": "function", "function": {"name": n}}),
                    )
                }
            }
            if let Some(s) = &case.schema {
                b.insert(
                    "response_format".into(),
                    json!({"type": "json_schema", "json_schema": {"name": "answer", "strict": true, "schema": s}}),
                );
            }
            if let Some(s) = &case.stop {
                b.insert("stop".into(), json!(s));
            }
            let key = if m.openai_reasoning {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            b.insert(key.into(), json!(mt));
            if temp {
                b.insert("temperature".into(), json!(0));
            }
            if case.reasoning {
                b.insert("reasoning_effort".into(), json!("low"));
            } else if m.openai_reasoning {
                b.insert("reasoning_effort".into(), json!("minimal"));
            }
            if case.stream {
                b.insert("stream".into(), json!(true));
                if case.include_usage {
                    b.insert("stream_options".into(), json!({"include_usage": true}));
                }
            }
        }
        Dialect::Responses => {
            let mut input = Vec::new();
            for msg in &case.msgs {
                match msg {
                    Msg::System(s) => _ = b.insert("instructions".into(), json!(s)),
                    Msg::User(parts) => input.push(json!({
                        "role": "user",
                        "content": parts.iter().map(|p| match p {
                            Part::Text(t) => json!({"type": "input_text", "text": t}),
                            Part::Png(data) => json!({"type": "input_image", "image_url": format!("data:image/png;base64,{data}")}),
                        }).collect::<Vec<_>>(),
                    })),
                    Msg::Assistant(s) => input.push(json!({"role": "assistant", "content": s})),
                    Msg::Calls(calls) => {
                        for (id, name, args) in calls {
                            input.push(json!({"type": "function_call", "call_id": id, "name": name, "arguments": args.to_string()}));
                        }
                    }
                    Msg::Results(rs) => {
                        for (id, content) in rs {
                            input.push(json!({"type": "function_call_output", "call_id": id, "output": content}));
                        }
                    }
                    Msg::BadRole(s) => input.push(json!({"role": "robot", "content": s})),
                }
            }
            b.insert("input".into(), json!(input));
            if case.tools {
                b.insert(
                    "tools".into(),
                    json!(
                        TOOLS
                            .iter()
                            .map(|(n, desc)| json!({
                                "type": "function", "name": n, "description": desc,
                                "parameters": tool_schema(n), "strict": false,
                            }))
                            .collect::<Vec<_>>()
                    ),
                );
            }
            match &case.choice {
                Choice::Unset => {}
                Choice::Auto => _ = b.insert("tool_choice".into(), json!("auto")),
                Choice::None => _ = b.insert("tool_choice".into(), json!("none")),
                Choice::Required => _ = b.insert("tool_choice".into(), json!("required")),
                Choice::Named(n) => {
                    _ = b.insert("tool_choice".into(), json!({"type": "function", "name": n}))
                }
            }
            if let Some(s) = &case.schema {
                b.insert(
                    "text".into(),
                    json!({"format": {"type": "json_schema", "name": "answer", "strict": true, "schema": s}}),
                );
            }
            b.insert("max_output_tokens".into(), json!(mt));
            if temp {
                b.insert("temperature".into(), json!(0));
            }
            if case.reasoning {
                b.insert("reasoning".into(), json!({"effort": "low"}));
            } else if m.openai_reasoning {
                b.insert("reasoning".into(), json!({"effort": "minimal"}));
            }
            if case.stream {
                b.insert("stream".into(), json!(true));
            }
        }
        Dialect::Messages => {
            let mut msgs = Vec::new();
            for msg in &case.msgs {
                match msg {
                    Msg::System(s) => _ = b.insert("system".into(), json!(s)),
                    Msg::User(parts) => msgs.push(json!({
                        "role": "user",
                        "content": parts.iter().map(|p| match p {
                            Part::Text(t) => json!({"type": "text", "text": t}),
                            Part::Png(data) => json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}),
                        }).collect::<Vec<_>>(),
                    })),
                    Msg::Assistant(s) => msgs.push(json!({"role": "assistant", "content": s})),
                    Msg::Calls(calls) => msgs.push(json!({
                        "role": "assistant",
                        "content": calls.iter().map(|(id, name, args)| json!({
                            "type": "tool_use", "id": id, "name": name, "input": args,
                        })).collect::<Vec<_>>(),
                    })),
                    Msg::Results(rs) => msgs.push(json!({
                        "role": "user",
                        "content": rs.iter().map(|(id, content)| json!({
                            "type": "tool_result", "tool_use_id": id, "content": content,
                        })).collect::<Vec<_>>(),
                    })),
                    Msg::BadRole(s) => msgs.push(json!({"role": "robot", "content": s})),
                }
            }
            b.insert("messages".into(), json!(msgs));
            if case.tools {
                b.insert(
                    "tools".into(),
                    json!(
                        TOOLS
                            .iter()
                            .map(|(n, desc)| json!({
                                "name": n, "description": desc, "input_schema": tool_schema(n),
                            }))
                            .collect::<Vec<_>>()
                    ),
                );
            }
            match &case.choice {
                Choice::Unset => {}
                Choice::Auto => _ = b.insert("tool_choice".into(), json!({"type": "auto"})),
                Choice::None => _ = b.insert("tool_choice".into(), json!({"type": "none"})),
                Choice::Required => _ = b.insert("tool_choice".into(), json!({"type": "any"})),
                Choice::Named(n) => {
                    _ = b.insert("tool_choice".into(), json!({"type": "tool", "name": n}))
                }
            }
            if let Some(s) = &case.schema {
                b.insert(
                    "output_config".into(),
                    json!({"format": {"type": "json_schema", "schema": s}}),
                );
            }
            if let Some(s) = &case.stop {
                b.insert("stop_sequences".into(), json!(s));
            }
            b.insert("max_tokens".into(), json!(mt));
            if temp {
                b.insert("temperature".into(), json!(0));
            }
            if case.reasoning {
                b.insert(
                    "thinking".into(),
                    json!({"type": "enabled", "budget_tokens": 1024}),
                );
            } else if m.openai_reasoning {
                b.insert("thinking".into(), json!({"type": "disabled"}));
            }
            if case.stream {
                b.insert("stream".into(), json!(true));
            }
        }
    }
    Value::Object(b)
}

fn chat_parts(parts: &[Part]) -> Value {
    if let [Part::Text(t)] = parts {
        return json!(t);
    }
    json!(parts.iter().map(|p| match p {
        Part::Text(t) => json!({"type": "text", "text": t}),
        Part::Png(data) => json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{data}")}}),
    }).collect::<Vec<_>>())
}

// --- A tiny PNG with a word on it ----------------------------------------------------------------

/// 5×7 glyphs, one row per entry, MSB = leftmost of 5 columns.
fn glyph(c: char) -> [u8; 7] {
    match c {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'J' => [7, 2, 2, 2, 2, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 17, 25, 21, 19, 17, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        _ => [0; 7],
    }
}

/// A grayscale PNG of `word` in black block letters on white, base64.
fn png_base64(word: &str) -> String {
    const SCALE: usize = 6;
    let cols = word.len() * 6 - 1 + 4;
    let (w, h) = (cols * SCALE, 11 * SCALE);
    let mut raw = Vec::with_capacity(h * (w + 1));
    for y in 0..h {
        raw.push(0); // filter: none
        let gy = y / SCALE;
        for x in 0..w {
            let gx = x / SCALE;
            let mut ink = false;
            if (2..9).contains(&gy) && gx >= 2 {
                let (ci, cx) = ((gx - 2) / 6, (gx - 2) % 6);
                if let Some(ch) = word.chars().nth(ci)
                    && cx < 5
                {
                    ink = glyph(ch)[gy - 2] & (16 >> cx) != 0;
                }
            }
            raw.push(if ink { 0 } else { 255 });
        }
    }
    // zlib with stored (uncompressed) deflate blocks.
    let mut z = vec![0x78, 0x01];
    let chunks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (i, c) in chunks.iter().enumerate() {
        z.push(u8::from(i + 1 == chunks.len()));
        let len = c.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(c);
    }
    let (mut a, mut bb) = (1u32, 0u32);
    for &x in &raw {
        a = (a + u32::from(x)) % 65_521;
        bb = (bb + a) % 65_521;
    }
    z.extend_from_slice(&((bb << 16) | a).to_be_bytes());
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
    for (ty, data) in [(b"IHDR", ihdr), (b"IDAT", z), (b"IEND", Vec::new())] {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut crc_in = ty.to_vec();
        crc_in.extend_from_slice(&data);
        png.extend_from_slice(&crc_in);
        png.extend_from_slice(&crc32(&crc_in).to_be_bytes());
    }
    base64(&png)
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

fn base64(data: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for ch in data.chunks(3) {
        let n = (u32::from(ch[0]) << 16)
            | (u32::from(*ch.get(1).unwrap_or(&0)) << 8)
            | u32::from(*ch.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= ch.len() {
                s.push(T[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

// --- Sending ------------------------------------------------------------------------------------

/// What came back from one call.
struct Obs {
    status: u16,
    content_type: String,
    request_id: Option<String>,
    /// Non-stream body (or an error body on a stream request).
    body: Option<Value>,
    /// SSE events `(event name, data, arrival in ms since the request)`.
    events: Vec<(String, Value, f64)>,
    raw: String,
}

/// [`post_once`], again after a transport failure (a TLS reset on connect) or a rate limit /
/// overload (429, 529): neither says anything about parity.
fn post(dir: &Path, tag: &str, url: &str, headers: &[String], body: &Value) -> Result<Obs, Failed> {
    let mut tries = 0;
    loop {
        tries += 1;
        let r = post_once(dir, tag, url, headers, body);
        let again = match &r {
            Err(_) => true,
            Ok(o) => matches!(o.status, 429 | 529),
        };
        if !again || tries == 4 {
            return r;
        }
        std::thread::sleep(Duration::from_secs(2 * tries));
    }
}

/// POST `body` to `url` with headers from a curl config on stdin (so no key is ever on a command
/// line), reading the answer line by line to time each SSE event.
fn post_once(
    dir: &Path,
    tag: &str,
    url: &str,
    headers: &[String],
    body: &Value,
) -> Result<Obs, Failed> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let body_path = dir.join(format!("{tag}.req.json"));
    let hdr_path = dir.join(format!("{tag}.headers"));
    std::fs::write(&body_path, body.to_string()).map_err(|e| e.to_string())?;
    let mut config = String::new();
    for h in headers {
        config.push_str(&format!("header = \"{}\"\n", h.replace('"', "\\\"")));
    }
    let start = Instant::now();
    let mut child = Command::new("curl")
        .args([
            "-sS",
            "-N",
            "--max-time",
            "180",
            "-K",
            "-",
            "-w",
            "\n%{http_code}",
        ])
        .arg("-D")
        .arg(&hdr_path)
        .args(["-H", "content-type: application/json", "--data-binary"])
        .arg(format!("@{}", body_path.display()))
        .arg(url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("curl: {e}"))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(config.as_bytes())
        .map_err(|e| format!("curl stdin: {e}"))?;
    let mut lines: Vec<(String, f64)> = Vec::new();
    let mut rd = BufReader::new(child.stdout.take().unwrap());
    loop {
        let mut line = String::new();
        match rd.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => lines.push((line, start.elapsed().as_secs_f64() * 1000.0)),
        }
    }
    let mut err = String::new();
    let _ = child.stderr.take().unwrap().read_to_string(&mut err);
    let _ = child.wait();
    let _ = std::fs::remove_file(&body_path);
    let status = lines
        .last()
        .and_then(|(l, _)| l.trim().parse::<u16>().ok())
        .unwrap_or(0);
    if status == 0 {
        return Err(format!("curl to {url} failed: {err}").into());
    }
    lines.pop();
    if let Some((l, _)) = lines.last_mut()
        && l.ends_with('\n')
    {
        l.pop();
    }
    let raw: String = lines.iter().map(|(l, _)| l.as_str()).collect();
    let hdrs = std::fs::read_to_string(&hdr_path).unwrap_or_default();
    let _ = std::fs::remove_file(&hdr_path);
    // The last header block (after any 100-continue).
    let block = hdrs
        .rsplit("\r\n\r\n")
        .find(|b| !b.trim().is_empty())
        .unwrap_or("");
    let header = |name: &str| {
        block
            .lines()
            .filter_map(|l| l.split_once(':'))
            .filter(|(k, _)| k.trim().eq_ignore_ascii_case(name))
            .map(|(_, v)| v.trim().to_owned())
            .next_back()
    };
    let content_type = header("content-type").unwrap_or_default();
    let request_id = header("x-beyond-request-id");
    let mut obs = Obs {
        status,
        content_type: content_type.clone(),
        request_id,
        body: None,
        events: Vec::new(),
        raw: raw.clone(),
    };
    if content_type.starts_with("text/event-stream") {
        let (mut name, mut data, mut t) = (String::new(), String::new(), 0.0);
        for (line, at) in &lines {
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !data.is_empty() {
                    let v = serde_json::from_str(&data).unwrap_or(Value::String(data.clone()));
                    obs.events.push((std::mem::take(&mut name), v, t));
                    data.clear();
                }
                name.clear();
            } else if let Some(v) = line.strip_prefix("event:") {
                name = v.trim().to_owned();
            } else if let Some(v) = line.strip_prefix("data:") {
                if data.is_empty() {
                    t = *at;
                }
                data.push_str(v.strip_prefix(' ').unwrap_or(v));
            }
        }
        if !data.is_empty() {
            let v = serde_json::from_str(&data).unwrap_or(Value::String(data.clone()));
            obs.events.push((name, v, t));
        }
    } else {
        obs.body = serde_json::from_str(&raw).ok();
    }
    Ok(obs)
}

fn direct_call(
    dir: &Path,
    tag: &str,
    d: Dialect,
    m: &Model,
    key: &str,
    body: &Value,
) -> Result<Obs, Failed> {
    let p = m.provider;
    let (url, headers) = match d {
        Dialect::Chat => (
            format!("{}/chat/completions", p.base),
            vec![format!("authorization: Bearer {key}")],
        ),
        Dialect::Responses => (
            format!("{}/responses", p.base),
            vec![format!("authorization: Bearer {key}")],
        ),
        Dialect::Messages => (
            format!("{}/messages", p.base),
            vec![
                format!("x-api-key: {key}"),
                "anthropic-version: 2023-06-01".to_owned(),
            ],
        ),
    };
    post(dir, tag, &url, &headers, body)
}

fn gateway_call(
    dir: &Path,
    tag: &str,
    gw: &Gateway,
    d: Dialect,
    body: &Value,
) -> Result<Obs, Failed> {
    let headers = match d {
        Dialect::Messages => vec![
            format!("x-api-key: {DEV_TOKEN}"),
            "anthropic-version: 2023-06-01".to_owned(),
        ],
        _ => vec![format!("authorization: Bearer {DEV_TOKEN}")],
    };
    let url = format!("http://127.0.0.1:{}{}", gw.port, d.gateway_path());
    post(dir, tag, &url, &headers, body)
}

// --- Reading an answer into a dialect-free summary -----------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Finish {
    Stop,
    Length,
    Tool,
    Filter,
    Other,
}

#[derive(Default, Debug)]
struct Usage {
    /// Whole prompt, cache included (Anthropic's three input fields summed).
    input: u64,
    output: u64,
    reasoning: u64,
}

#[derive(Default)]
struct Summary {
    status: u16,
    content_type: String,
    /// `(type, code)` from the error envelope.
    error: Option<(String, String)>,
    /// The envelope is the dialect's own (`{"error": {...}}` / `{"type": "error", ...}`).
    error_envelope_ok: bool,
    finish_raw: String,
    finish: Option<Finish>,
    text: String,
    /// `(name, arguments as sent)`.
    calls: Vec<(String, String)>,
    usage: Option<Usage>,
    /// Non-stream: the body's type skeleton.
    shape: BTreeSet<String>,
    /// Stream: event types with repeats collapsed.
    seq: Vec<String>,
    /// Stream: each event type's keys (top level, plus a Chat chunk's delta keys).
    keys: BTreeMap<String, BTreeSet<String>>,
    /// Stream: first-to-last event, ms.
    span_ms: f64,
    events: usize,
}

/// Every `path:type` in a value; arrays collapse their elements, and `type` / `object` / `role`
/// discriminants name the element so content-block and item types are part of the shape.
fn skeleton(v: &Value, path: &str, out: &mut BTreeSet<String>) {
    match v {
        Value::Object(m) => {
            out.insert(format!("{path}:object"));
            for (k, x) in m {
                let p = format!("{path}.{k}");
                if matches!(k.as_str(), "type" | "object" | "role")
                    && let Some(s) = x.as_str()
                {
                    out.insert(format!("{p}={s}"));
                } else {
                    skeleton(x, &p, out);
                }
            }
        }
        Value::Array(a) => {
            out.insert(format!("{path}:array"));
            for x in a {
                let tag = x
                    .get("type")
                    .and_then(Value::as_str)
                    .map(|t| format!("[{t}]"))
                    .unwrap_or_else(|| "[]".to_owned());
                skeleton(x, &format!("{path}{tag}"), out);
            }
        }
        Value::Null => _ = out.insert(format!("{path}:null")),
        Value::Bool(_) => _ = out.insert(format!("{path}:bool")),
        Value::Number(_) => _ = out.insert(format!("{path}:number")),
        Value::String(_) => _ = out.insert(format!("{path}:string")),
    }
}

fn n(v: &Value) -> u64 {
    v.as_u64().unwrap_or(0)
}

fn chat_finish(s: &str) -> Finish {
    match s {
        "stop" => Finish::Stop,
        "length" => Finish::Length,
        "tool_calls" | "function_call" => Finish::Tool,
        "content_filter" => Finish::Filter,
        _ => Finish::Other,
    }
}

fn messages_finish(s: &str) -> Finish {
    match s {
        "end_turn" | "stop_sequence" => Finish::Stop,
        "max_tokens" | "model_context_window_exceeded" => Finish::Length,
        "tool_use" => Finish::Tool,
        "refusal" => Finish::Filter,
        _ => Finish::Other,
    }
}

fn chat_usage(u: &Value) -> Option<Usage> {
    u.get("prompt_tokens").map(|_| Usage {
        input: n(&u["prompt_tokens"]),
        output: n(&u["completion_tokens"]),
        reasoning: n(&u["completion_tokens_details"]["reasoning_tokens"]),
    })
}

fn responses_usage(u: &Value) -> Option<Usage> {
    u.get("input_tokens").map(|_| Usage {
        input: n(&u["input_tokens"]),
        output: n(&u["output_tokens"]),
        reasoning: n(&u["output_tokens_details"]["reasoning_tokens"]),
    })
}

fn messages_usage(u: &Value) -> Option<Usage> {
    u.get("input_tokens").map(|_| Usage {
        input: n(&u["input_tokens"])
            + n(&u["cache_read_input_tokens"])
            + n(&u["cache_creation_input_tokens"]),
        output: n(&u["output_tokens"]),
        reasoning: 0,
    })
}

fn error_of(d: Dialect, v: &Value) -> Option<(String, String, bool)> {
    let e = v.get("error")?;
    if e.is_null() {
        return None;
    }
    let s = |x: &Value| match x {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    let envelope_ok = match d {
        Dialect::Messages => v["type"] == "error" && e.get("type").is_some_and(Value::is_string),
        _ => e.is_object() && e.get("message").is_some_and(Value::is_string),
    };
    Some((s(&e["type"]), s(&e["code"]), envelope_ok))
}

/// Responses `output` items: text, calls, finish.
fn responses_output(resp: &Value, s: &mut Summary) {
    for item in resp["output"].as_array().into_iter().flatten() {
        match item["type"].as_str() {
            Some("message") => {
                for part in item["content"].as_array().into_iter().flatten() {
                    if let Some(t) = part["text"].as_str() {
                        s.text.push_str(t);
                    }
                    if let Some(t) = part["refusal"].as_str() {
                        s.text.push_str(t);
                    }
                }
            }
            Some("function_call") => s.calls.push((
                item["name"].as_str().unwrap_or("").to_owned(),
                item["arguments"].as_str().unwrap_or("").to_owned(),
            )),
            _ => {}
        }
    }
    let status = resp["status"].as_str().unwrap_or("");
    let reason = resp["incomplete_details"]["reason"].as_str().unwrap_or("");
    s.finish_raw = if reason.is_empty() {
        status.to_owned()
    } else {
        format!("{status}:{reason}")
    };
    s.finish = Some(match (status, reason) {
        ("completed", _) if !s.calls.is_empty() => Finish::Tool,
        ("completed", _) => Finish::Stop,
        ("incomplete", "max_output_tokens") => Finish::Length,
        ("incomplete", "content_filter") => Finish::Filter,
        _ => Finish::Other,
    });
    s.usage = responses_usage(&resp["usage"]);
}

fn summarize(d: Dialect, o: &Obs, include_usage_requested: bool) -> Summary {
    let mut s = Summary {
        status: o.status,
        content_type: o
            .content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_owned(),
        ..Summary::default()
    };
    if let Some(body) = &o.body {
        if let Some((t, c, ok)) = error_of(d, body) {
            s.error = Some((t, c));
            s.error_envelope_ok = ok;
            skeleton(body, "$", &mut s.shape);
            return s;
        }
        skeleton(body, "$", &mut s.shape);
        match d {
            Dialect::Chat => {
                let ch = &body["choices"][0];
                s.text = ch["message"]["content"].as_str().unwrap_or("").to_owned();
                if let Some(r) = ch["message"]["refusal"].as_str() {
                    s.text.push_str(r);
                }
                for tc in ch["message"]["tool_calls"].as_array().into_iter().flatten() {
                    s.calls.push((
                        tc["function"]["name"].as_str().unwrap_or("").to_owned(),
                        tc["function"]["arguments"]
                            .as_str()
                            .unwrap_or("")
                            .to_owned(),
                    ));
                }
                s.finish_raw = ch["finish_reason"].as_str().unwrap_or("").to_owned();
                s.usage = chat_usage(&body["usage"]);
            }
            Dialect::Responses => responses_output(body, &mut s),
            Dialect::Messages => {
                for b in body["content"].as_array().into_iter().flatten() {
                    match b["type"].as_str() {
                        Some("text") => s.text.push_str(b["text"].as_str().unwrap_or("")),
                        Some("tool_use") => s.calls.push((
                            b["name"].as_str().unwrap_or("").to_owned(),
                            b["input"].to_string(),
                        )),
                        _ => {}
                    }
                }
                s.finish_raw = body["stop_reason"].as_str().unwrap_or("").to_owned();
                s.usage = messages_usage(&body["usage"]);
            }
        }
    } else if !o.events.is_empty() {
        let mut types = Vec::new();
        let first = o.events.first().map_or(0.0, |e| e.2);
        let last = o.events.last().map_or(0.0, |e| e.2);
        s.span_ms = last - first;
        s.events = o.events.len();
        // Chat: tool-call fragments by index.
        let mut chat_calls: BTreeMap<u64, (String, String)> = BTreeMap::new();
        // Messages: tool_use blocks by index.
        let mut msg_calls: BTreeMap<u64, (String, String)> = BTreeMap::new();
        let mut msg_usage = Usage::default();
        let mut msg_usage_seen = false;
        let mut role_seen = false;
        for (name, data, _) in &o.events {
            let ty = match d {
                Dialect::Chat => {
                    if data.as_str() == Some("[DONE]") {
                        "done".to_owned()
                    } else if let Some((t, c, ok)) = error_of(d, data) {
                        s.error = Some((t, c));
                        s.error_envelope_ok = ok;
                        "error".to_owned()
                    } else {
                        let ch = &data["choices"][0];
                        let delta = &ch["delta"];
                        let mut flags = Vec::new();
                        // OpenRouter repeats `role` on every chunk and the gateway drops the
                        // repeats (`ChatIdentity`, documented): only the first one counts.
                        if delta.get("role").is_some_and(|r| !r.is_null()) && !role_seen {
                            role_seen = true;
                            flags.push("role");
                        }
                        if delta["content"].as_str().is_some_and(|c| !c.is_empty()) {
                            s.text.push_str(delta["content"].as_str().unwrap());
                            flags.push("content");
                        }
                        if delta["refusal"].as_str().is_some_and(|c| !c.is_empty()) {
                            s.text.push_str(delta["refusal"].as_str().unwrap());
                            flags.push("refusal");
                        }
                        let reasoning = ["reasoning", "reasoning_content"]
                            .iter()
                            .any(|k| delta[*k].as_str().is_some_and(|c| !c.is_empty()))
                            || delta
                                .get("reasoning_details")
                                .and_then(Value::as_array)
                                .is_some_and(|a| !a.is_empty());
                        if reasoning {
                            flags.push("reasoning");
                        }
                        if let Some(tcs) = delta["tool_calls"].as_array()
                            && !tcs.is_empty()
                        {
                            flags.push("tool_calls");
                            for tc in tcs {
                                let e = chat_calls.entry(n(&tc["index"])).or_default();
                                if let Some(nm) = tc["function"]["name"].as_str() {
                                    e.0.push_str(nm);
                                }
                                if let Some(a) = tc["function"]["arguments"].as_str() {
                                    e.1.push_str(a);
                                }
                            }
                        }
                        if let Some(f) = ch["finish_reason"].as_str() {
                            s.finish_raw = f.to_owned();
                            flags.push("finish");
                        }
                        if let Some(u) = chat_usage(&data["usage"]) {
                            s.usage = Some(u);
                            flags.push("usage");
                        }
                        if flags.is_empty() {
                            flags.push("empty");
                        }
                        let keys = s.keys.entry("chunk".to_owned()).or_default();
                        for k in data.as_object().into_iter().flat_map(|m| m.keys()) {
                            if k != "usage" || include_usage_requested {
                                keys.insert(k.clone());
                            }
                        }
                        for k in delta.as_object().into_iter().flat_map(|m| m.keys()) {
                            keys.insert(format!("delta.{k}"));
                        }
                        flags.join("+")
                    }
                }
                Dialect::Messages => {
                    let t = if name.is_empty() {
                        data["type"].as_str().unwrap_or("").to_owned()
                    } else {
                        name.clone()
                    };
                    let idx = n(&data["index"]);
                    let sub = match t.as_str() {
                        "message_start" => {
                            if let Some(u) = data["message"].get("usage") {
                                msg_usage_seen = true;
                                msg_usage = messages_usage(u).unwrap_or_default();
                            }
                            String::new()
                        }
                        "content_block_start" => {
                            let b = &data["content_block"];
                            if b["type"] == "tool_use" {
                                let input = if b["input"].as_object().is_some_and(|m| !m.is_empty())
                                {
                                    b["input"].to_string()
                                } else {
                                    String::new()
                                };
                                msg_calls.insert(
                                    idx,
                                    (b["name"].as_str().unwrap_or("").to_owned(), input),
                                );
                            }
                            if let Some(t) = b["text"].as_str() {
                                s.text.push_str(t);
                            }
                            b["type"].as_str().unwrap_or("").to_owned()
                        }
                        "content_block_delta" => {
                            let dl = &data["delta"];
                            match dl["type"].as_str() {
                                Some("text_delta") => {
                                    s.text.push_str(dl["text"].as_str().unwrap_or(""))
                                }
                                Some("input_json_delta") => {
                                    if let Some(c) = msg_calls.get_mut(&idx) {
                                        c.1.push_str(dl["partial_json"].as_str().unwrap_or(""));
                                    }
                                }
                                _ => {}
                            }
                            dl["type"].as_str().unwrap_or("").to_owned()
                        }
                        "message_delta" => {
                            if let Some(r) = data["delta"]["stop_reason"].as_str() {
                                s.finish_raw = r.to_owned();
                            }
                            let u = &data["usage"];
                            if u.is_object() {
                                msg_usage_seen = true;
                                if u.get("output_tokens").is_some() {
                                    msg_usage.output = n(&u["output_tokens"]);
                                }
                                if u.get("input_tokens").is_some() {
                                    msg_usage.input = n(&u["input_tokens"])
                                        + n(&u["cache_read_input_tokens"])
                                        + n(&u["cache_creation_input_tokens"]);
                                }
                            }
                            String::new()
                        }
                        "error" => {
                            if let Some((et, c, _)) = error_of(d, data) {
                                s.error = Some((et, c));
                                s.error_envelope_ok = true;
                            }
                            String::new()
                        }
                        _ => String::new(),
                    };
                    if t == "ping" {
                        continue;
                    }
                    let keys = s.keys.entry(t.clone()).or_default();
                    for k in data.as_object().into_iter().flat_map(|m| m.keys()) {
                        keys.insert(k.clone());
                    }
                    if sub.is_empty() {
                        t
                    } else {
                        format!("{t}:{sub}")
                    }
                }
                Dialect::Responses => {
                    let t = data["type"].as_str().unwrap_or(name.as_str()).to_owned();
                    if matches!(
                        t.as_str(),
                        "response.completed" | "response.incomplete" | "response.failed"
                    ) {
                        responses_output(&data["response"], &mut s);
                        if t == "response.failed"
                            && let Some((et, c, _)) = error_of(d, &data["response"])
                        {
                            s.error = Some((et, c));
                        }
                    }
                    if t == "error" {
                        let e = data.get("error").unwrap_or(data);
                        s.error = Some((
                            e["type"].as_str().unwrap_or("").to_owned(),
                            e["code"].as_str().unwrap_or("").to_owned(),
                        ));
                        s.error_envelope_ok = true;
                    }
                    let keys = s.keys.entry(t.clone()).or_default();
                    for k in data.as_object().into_iter().flat_map(|m| m.keys()) {
                        keys.insert(k.clone());
                    }
                    let item = data["item"]["type"]
                        .as_str()
                        .or(data["part"]["type"].as_str());
                    match item {
                        Some(it) if t.contains("output_item") || t.contains("content_part") => {
                            format!("{t}:{it}")
                        }
                        _ => t,
                    }
                }
            };
            types.push(ty);
        }
        if d == Dialect::Chat {
            s.calls = chat_calls.into_values().collect();
        }
        if d == Dialect::Messages {
            s.calls = msg_calls
                .into_values()
                .map(|(nm, a)| (nm, if a.is_empty() { "{}".to_owned() } else { a }))
                .collect();
            if msg_usage_seen {
                s.usage = Some(msg_usage);
            }
        }
        types.dedup();
        s.seq = types;
    } else if o.status >= 400 {
        s.error = Some(("<non-json>".to_owned(), String::new()));
    }
    if s.finish.is_none() && !s.finish_raw.is_empty() {
        s.finish = Some(match d {
            Dialect::Messages => messages_finish(&s.finish_raw),
            _ => chat_finish(&s.finish_raw),
        });
    }
    // Calls under a plain stop are still a tool turn (the gateway's own rule).
    if !s.calls.is_empty() && s.finish == Some(Finish::Stop) {
        s.finish = Some(Finish::Tool);
    }
    s
}

// --- Checks ---------------------------------------------------------------------------------------

/// A JSON-schema subset: object/array/string/integer/number/boolean, properties, required,
/// additionalProperties: false, items.
fn validate(v: &Value, schema: &Value) -> bool {
    match schema["type"].as_str() {
        Some("object") => {
            let Some(m) = v.as_object() else { return false };
            let props = schema["properties"].as_object();
            for r in schema["required"].as_array().into_iter().flatten() {
                if !m.contains_key(r.as_str().unwrap_or("")) {
                    return false;
                }
            }
            for (k, x) in m {
                match props.and_then(|p| p.get(k)) {
                    Some(ps) => {
                        if !validate(x, ps) {
                            return false;
                        }
                    }
                    None if schema["additionalProperties"] == false => return false,
                    None => {}
                }
            }
            true
        }
        Some("array") => v
            .as_array()
            .is_some_and(|a| a.iter().all(|x| validate(x, &schema["items"]))),
        Some("string") => v.is_string(),
        Some("integer") => v.is_i64() || v.is_u64(),
        Some("number") => v.is_number(),
        Some("boolean") => v.is_boolean(),
        _ => true,
    }
}

fn args_valid(calls: &[(String, String)]) -> bool {
    calls.iter().all(|(name, args)| {
        TOOLS.iter().any(|(t, _)| t == name)
            && serde_json::from_str::<Value>(args).is_ok_and(|v| validate(&v, &tool_schema(name)))
    })
}

fn has_all(text: &str, words: &[String]) -> bool {
    let t = text.to_lowercase();
    words.iter().all(|w| t.contains(&w.to_lowercase()))
}

fn has_any(text: &str, words: &[String]) -> bool {
    let t = text.to_lowercase();
    words.iter().any(|w| t.contains(&w.to_lowercase()))
}

fn names(calls: &[(String, String)]) -> BTreeSet<&str> {
    calls.iter().map(|(n, _)| n.as_str()).collect()
}

fn structured_ok(text: &str, schema: &Value) -> bool {
    let t = text.trim();
    let t = t
        .strip_prefix("```json")
        .or_else(|| t.strip_prefix("```"))
        .map_or(t, |x| x.trim_end_matches("```").trim());
    serde_json::from_str::<Value>(t).is_ok_and(|v| validate(&v, schema))
}

/// Shape entries that may differ on a same-dialect relay. No documented gateway change shows in a
/// non-stream body; the only exemption is model nondeterminism measured direct-vs-direct: whether
/// grok returns its `reasoning_content` varies call to call.
fn allowed_shape(m: &Model, entry: &str) -> bool {
    m.always_reasons && entry.contains("reasoning_content")
}

/// Everything about `gw` that disagrees with `direct` for `case` on `path`.
fn compare(path: &ParityPath, case: &Case, direct: &Summary, gw: &Summary) -> Vec<String> {
    let mut p = Vec::new();
    let cross = path.cross();
    if direct.status / 100 != gw.status / 100 {
        p.push(format!(
            "HTTP status class: direct {} gateway {}",
            direct.status, gw.status
        ));
        return p;
    }
    match (&direct.error, &gw.error) {
        (Some(de), Some(ge)) => {
            // Cross dialect: the client's own envelope. Same dialect: the provider's, whatever it
            // is (xAI's isn't OpenAI's).
            let envelope_bad = if cross {
                !gw.error_envelope_ok
            } else {
                direct.error_envelope_ok != gw.error_envelope_ok
            };
            if envelope_bad {
                p.push(format!(
                    "gateway error isn't in the {:?} client's envelope",
                    path.client
                ));
            }
            if !cross && de != ge {
                p.push(format!("error type/code: direct {de:?} gateway {ge:?}"));
            }
            return p;
        }
        (Some(de), None) => {
            p.push(format!("direct errored {de:?}, gateway did not"));
            return p;
        }
        (None, Some(ge)) => {
            p.push(format!("gateway errored {ge:?}, direct did not"));
            return p;
        }
        (None, None) => {}
    }
    if direct.finish != gw.finish {
        p.push(format!(
            "finish class: direct {:?} gateway {:?}",
            direct.finish, gw.finish
        ));
    }
    if names(&direct.calls) != names(&gw.calls) {
        p.push(format!(
            "tool names: direct {:?} gateway {:?}",
            names(&direct.calls),
            names(&gw.calls)
        ));
    }
    if case.kind == Kind::ParallelTools && (direct.calls.len() >= 2) != (gw.calls.len() >= 2) {
        p.push(format!(
            "parallel calls: direct {} gateway {}",
            direct.calls.len(),
            gw.calls.len()
        ));
    }
    if args_valid(&direct.calls) && !args_valid(&gw.calls) {
        p.push(format!("tool arguments invalid at gateway: {:?}", gw.calls));
    }
    if let Some(schema) = &case.schema
        && structured_ok(&direct.text, schema)
        && !structured_ok(&gw.text, schema)
    {
        p.push(format!(
            "structured output invalid at gateway: {:?}",
            gw.text
        ));
    }
    if !case.expect.is_empty()
        && has_all(&direct.text, &case.expect) != has_all(&gw.text, &case.expect)
    {
        p.push(format!(
            "expected {:?} in text: direct {} gateway {}",
            case.expect,
            has_all(&direct.text, &case.expect),
            has_all(&gw.text, &case.expect)
        ));
    }
    if !case.absent.is_empty()
        && has_any(&direct.text, &case.absent) != has_any(&gw.text, &case.absent)
    {
        p.push(format!(
            "text past the stop sequence: direct {:?} gateway {:?}",
            direct.text, gw.text
        ));
    }
    // Usage: the gateway always meters; a Chat stream without include_usage has no usage chunk
    // at the provider, and the gateway's injected one is documented.
    let direct_usage_expected =
        !(case.stream && path.direct_dialect() == Dialect::Chat && !case.include_usage);
    if direct.usage.is_some() && gw.usage.is_none() {
        p.push("usage missing at gateway".to_owned());
    }
    if direct_usage_expected && direct.usage.is_none() && gw.usage.is_some() && !cross {
        p.push("usage at gateway but not direct".to_owned());
    }
    if cross && let (Some(du), Some(gu)) = (&direct.usage, &gw.usage) {
        let (a, b) = (du.input as f64, gu.input as f64);
        if (a - b).abs() > (a * 0.35).max(60.0) {
            p.push(format!(
                "input tokens: direct {} gateway {}",
                du.input, gu.input
            ));
        }
    }
    if !cross {
        if direct.content_type != gw.content_type {
            p.push(format!(
                "content-type: direct {} gateway {}",
                direct.content_type, gw.content_type
            ));
        }
        if direct.finish_raw != gw.finish_raw {
            p.push(format!(
                "finish reason: direct {:?} gateway {:?}",
                direct.finish_raw, gw.finish_raw
            ));
        }
        let missing: Vec<&String> = direct
            .shape
            .difference(&gw.shape)
            .filter(|e| !allowed_shape(&path.model, e))
            .collect();
        let extra: Vec<&String> = gw
            .shape
            .difference(&direct.shape)
            .filter(|e| !allowed_shape(&path.model, e))
            .collect();
        if !missing.is_empty() || !extra.is_empty() {
            p.push(format!(
                "shape: missing at gateway {missing:?}, extra at gateway {extra:?}"
            ));
        }
        if case.stream {
            let strip = |seq: &[String]| -> Vec<String> {
                let keep = |f: &&str| {
                    // The injected usage chunk is documented.
                    (case.include_usage || *f != "usage")
                        // Whether grok streams its reasoning varies call to call, direct too.
                        && (!path.model.always_reasons || *f != "reasoning")
                };
                let mut v: Vec<String> = if path.model.always_reasons {
                    // ... and so does which deltas share a chunk with it: compare the delta kinds'
                    // order, not their packing.
                    seq.iter()
                        .flat_map(|t| t.split('+'))
                        .filter(keep)
                        .map(str::to_owned)
                        .collect()
                } else {
                    seq.iter()
                        .map(|t| t.split('+').filter(keep).collect::<Vec<_>>().join("+"))
                        .filter(|t| !t.is_empty())
                        .collect()
                };
                v.dedup();
                v
            };
            let (ds, gs) = (strip(&direct.seq), strip(&gw.seq));
            if ds != gs {
                p.push(format!(
                    "stream event sequence: direct {ds:?} gateway {gs:?}"
                ));
            }
            for (t, dk) in &direct.keys {
                let gk = gw.keys.get(t).cloned().unwrap_or_default();
                let volatile =
                    |k: &&String| path.model.always_reasons && k.starts_with("delta.reasoning");
                let missing: Vec<&String> = dk.difference(&gk).filter(|k| !volatile(k)).collect();
                let extra: Vec<&String> = gk.difference(dk).filter(|k| !volatile(k)).collect();
                if gw.keys.contains_key(t) && (!missing.is_empty() || !extra.is_empty()) {
                    p.push(format!(
                        "stream {t} keys: missing at gateway {missing:?}, extra {extra:?}"
                    ));
                }
            }
        }
    }
    p
}

/// S1: a stream that took the provider a while arrives spread out through the gateway too.
fn incremental(direct: &Summary, gw: &Summary) -> Option<String> {
    if direct.span_ms >= 400.0 && direct.events >= 4 && gw.span_ms < direct.span_ms * 0.2 {
        return Some(format!(
            "stream buffered: direct spread {:.0} ms over {} events, gateway {:.0} ms over {}",
            direct.span_ms, direct.events, gw.span_ms, gw.events
        ));
    }
    None
}

/// B1: exactly one ledger row for the gateway call, from the expected provider, with the client's
/// tokens.
fn ledger(gwy: &Gateway, path: &ParityPath, obs: &Obs, s: &Summary) -> Vec<String> {
    let mut p = Vec::new();
    let Some(id) = &obs.request_id else {
        return vec!["gateway answer carried no x-beyond-request-id".to_owned()];
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    let rows = loop {
        let rows: Vec<Value> = usage_rows(&gwy.log)
            .into_iter()
            .filter(|r| r["request_id"] == id.as_str())
            .collect();
        if !rows.is_empty() || Instant::now() >= deadline {
            break rows;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let [row] = rows.as_slice() else {
        return vec![format!(
            "{id}: {} ai.usage rows, want exactly 1",
            rows.len()
        )];
    };
    if row["provider"] != path.model.provider.name {
        p.push(format!(
            "{id}: served by {}, want {}",
            row["provider"], path.model.provider.name
        ));
    }
    if row["usage_estimated"] == true {
        p.push(format!("{id}: row is an estimate ({row})"));
    }
    if let Some(u) = &s.usage {
        let anthropic = match row["usage_wire"].as_str() {
            Some(w) => w == "anthropic",
            None => matches!(row["provider"].as_str(), Some("anthropic" | "bedrock")),
        };
        let row_input = n(&row["input_tokens"])
            + if anthropic {
                n(&row["cache_read_tokens"]) + n(&row["cache_write_tokens"])
            } else {
                0
            };
        if row_input != u.input {
            p.push(format!(
                "{id}: client saw {} input tokens, row bills {row_input}",
                u.input
            ));
        }
        let out = n(&row["output_tokens"]);
        if out != u.output && out != u.output + u.reasoning {
            p.push(format!(
                "{id}: client saw {} output tokens (+{} reasoning), row bills {out}",
                u.output, u.reasoning
            ));
        }
    } else {
        p.push(format!("{id}: the client was shown no usage"));
    }
    p
}

// --- A trial -------------------------------------------------------------------------------------

fn claims_for(path: &ParityPath, case: &Case) -> String {
    // An invalid request proves the error path (T6), not the endpoint working.
    let mut c: Vec<&str> = if case.kind.is_error() {
        Vec::new()
    } else {
        let mut c = path.claims.to_vec();
        c.push("B1");
        c
    };
    if case.kind.is_tools() && !case.kind.is_error() {
        c.push("T1");
    }
    match case.kind {
        Kind::Image => c.push("T3"),
        Kind::JsonSchema => c.push("T4"),
        Kind::ErrorRole | Kind::ErrorToolChoice => c.push("T6"),
        _ => {}
    }
    if case.stream && !case.kind.is_error() {
        c.push("S1");
    }
    if path.cross() {
        if !case.kind.is_error() {
            c.extend(["TRN-22", "TRN-23"]);
        }
        match case.kind {
            Kind::System => c.push("TRN-2"),
            // Translation must not turn a request the provider rejects into one it accepts.
            Kind::ErrorRole | Kind::ErrorToolChoice => c.push("TRN-21"),
            Kind::Reasoning => c.push("T5"),
            Kind::Text if path.client == Dialect::Responses => c.push("TRN-1"),
            Kind::ParallelTools if case.stream && path.client == Dialect::Messages => {
                c.push("TRN-12")
            }
            _ => {}
        }
    }
    let mut seen = BTreeSet::new();
    c.retain(|x| seen.insert(*x));
    c.join("+")
}

fn fragment(s: &Summary) -> String {
    let text: String = s.text.chars().take(160).collect();
    format!(
        "status={} error={:?} finish={:?}/{:?} calls={:?} text={text:?} usage={:?} seq={:?}",
        s.status, s.error, s.finish, s.finish_raw, s.calls, s.usage, s.seq
    )
}

fn price_of(m: &Model, u: &Option<Usage>, provider: &str) -> f64 {
    let Some(u) = u else { return 0.0 };
    let out = u.output + if provider == "xai" { u.reasoning } else { 0 };
    (u.input as f64 * m.price.0 + out as f64 * m.price.1) / 1e6
}

fn run_trial(
    name: &str,
    path: ParityPath,
    case: &Case,
    keys: &BTreeMap<String, String>,
) -> Result<(), Failed> {
    let dir = scratch().join(name.replace("::", "__").replace('+', "_"));
    let gwy = gateway(path.model.provider, &keys[path.model.provider.var])?;
    let key = &keys[path.model.provider.var];
    let dd = path.direct_dialect();
    let direct_body = render(case, dd, &path.model, path.model.direct);
    let gw_body = render(case, path.client, &path.model, path.model.gw);
    let mut cost = 0.0;
    let mut first: Option<Vec<String>> = None;
    for attempt in 0..2 {
        let d_obs = direct_call(
            &dir,
            &format!("direct{attempt}"),
            dd,
            &path.model,
            key,
            &direct_body,
        )?;
        let g_obs = gateway_call(
            &dir,
            &format!("gateway{attempt}"),
            &gwy,
            path.client,
            &gw_body,
        )?;
        let ds = summarize(dd, &d_obs, case.include_usage);
        let gs = summarize(path.client, &g_obs, case.include_usage);
        cost += price_of(&path.model, &ds.usage, path.model.provider.name)
            + price_of(&path.model, &gs.usage, path.model.provider.name);
        let mut problems = compare(&path, case, &ds, &gs);
        if case.stream && problems.is_empty() && ds.error.is_none() {
            problems.extend(incremental(&ds, &gs));
        }
        if std::env::var_os("PARITY_DUMP").is_some() {
            eprintln!(
                "{name} attempt {attempt}:\n  direct:  {}\n  gateway: {}\n  problems: {problems:?}",
                fragment(&ds),
                fragment(&gs)
            );
        }
        if problems.is_empty() {
            let mut lp = Vec::new();
            if gs.error.is_none() && gs.status < 400 {
                lp = ledger(&gwy, &path, &g_obs, &gs);
            }
            record_cost(name, cost);
            if let Some(f) = first {
                eprintln!(
                    "parity: first attempt disagreed (nondeterminism?): {}",
                    f.join("; ")
                );
            }
            if lp.is_empty() {
                let _ = std::fs::remove_dir_all(&dir);
                return Ok(());
            }
            return Err(format!("ledger disagrees with the client:\n  {}", lp.join("\n  ")).into());
        }
        if attempt == 0 {
            first = Some(problems);
            continue;
        }
        record_cost(name, cost);
        std::fs::create_dir_all(&dir).ok();
        let _ = std::fs::write(dir.join("direct.req.json"), direct_body.to_string());
        let _ = std::fs::write(dir.join("gateway.req.json"), gw_body.to_string());
        let _ = std::fs::write(dir.join("direct.out"), &d_obs.raw);
        let _ = std::fs::write(dir.join("gateway.out"), &g_obs.raw);
        return Err(format!(
            "parity mismatch (twice):\n  attempt 1: {}\n  attempt 2: {}\n  direct ({dd:?}):  {}\n  gateway ({:?}): {}\n  raw answers in {}",
            first.unwrap_or_default().join("; "),
            problems.join("; "),
            fragment(&ds),
            path.client,
            fragment(&gs),
            dir.display()
        )
        .into());
    }
    unreachable!()
}

fn scratch() -> PathBuf {
    repo_root().join("target/verify-parity")
}

fn record_cost(name: &str, usd: f64) {
    *COST.lock().unwrap() += usd;
    let _ = std::fs::create_dir_all(scratch());
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(scratch().join("costs.jsonl"))
    {
        // One write per line: trials append concurrently.
        let _ = f.write_all(format!("{}\n", json!({"trial": name, "usd": usd})).as_bytes());
    }
}

static COST: Mutex<f64> = Mutex::new(0.0);

/// A rough per-trial estimate: two calls, prompt ~350 tokens (+ tools, image), answer by model.
fn estimate(path: &ParityPath, case: &Case) -> f64 {
    let mut input = 350.0;
    if case.tools {
        input += 200.0;
    }
    if case.kind == Kind::Image {
        input += 150.0;
    }
    let out = if case.reasoning {
        900.0
    } else if path.model.always_reasons {
        600.0
    } else if path.model.openai_reasoning {
        250.0
    } else {
        60.0
    };
    2.0 * (input * path.model.price.0 + out * path.model.price.1) / 1e6
}

// --- Gateways ------------------------------------------------------------------------------------

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
    out.extend(std::env::vars());
    out.retain(|_, v| !v.is_empty());
    out
}

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

/// A gateway holding one provider's pool key (so no walk can fail over or probe another
/// provider), and its nats-server; both killed on drop.
struct Gateway {
    _nats: Guard,
    _gw: Guard,
    port: u16,
    log: PathBuf,
}

struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

type Gateways = Mutex<BTreeMap<&'static str, Arc<Gateway>>>;

fn gateways() -> &'static Gateways {
    static G: OnceLock<Gateways> = OnceLock::new();
    G.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The shared gateway for `provider`, booted on first use (one per process: nextest runs each
/// trial in its own).
fn gateway(provider: Provider, pool: &str) -> Result<Arc<Gateway>, Failed> {
    let mut g = gateways().lock().unwrap();
    if let Some(gw) = g.get(provider.name) {
        return Ok(gw.clone());
    }
    let gw = Arc::new(boot(provider.name, pool)?);
    g.insert(provider.name, gw.clone());
    Ok(gw)
}

/// A free port below the kernel's ephemeral range. Other suites on this host take theirs from
/// `bind(0)` (the ephemeral range) and release them before their gateway binds; Pingora binds
/// with `SO_REUSEPORT`, so a port two gateways both picked is shared silently and some requests
/// land on the other one (seen once: a 404 from a stranger's admin listener).
fn free_port() -> u16 {
    static NEXT: Mutex<u64> = Mutex::new(0);
    let mut seed = NEXT.lock().unwrap();
    if *seed == 0 {
        *seed = u64::from(std::process::id())
            ^ std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
    }
    loop {
        let port = 20_000 + (Rng(*seed).next() % 12_000) as u16;
        *seed = seed.wrapping_add(1);
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
}

fn boot(provider: &str, pool: &str) -> Result<Gateway, Failed> {
    let dir = scratch().join(format!(
        "gw-{provider}-{}-{}",
        std::process::id(),
        free_port()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let nats_port = free_port();
    let nats = Guard(
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
    let cfg = format!(
        "listen = \"127.0.0.1:{port}\"\nmetrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\n\
         upstream_tls = true\n\n[pool_keys]\n{provider} = [{pool:?}]\n\n\
         [signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"
    );
    let cfg_path = dir.join("gateway.toml");
    std::fs::write(&cfg_path, cfg).map_err(|e| e.to_string())?;
    let log = dir.join("gateway.log");
    let file = std::fs::File::create(&log).map_err(|e| e.to_string())?;
    let mut gw = Guard(
        Command::new(gateway_bin())
            .args(["run", "-c"])
            .arg(&cfg_path)
            .env("AI_LOG", "warn,ai.usage=info")
            .stdout(file.try_clone().map_err(|e| e.to_string())?)
            .stderr(file)
            .spawn()
            .map_err(|e| format!("gateway: {e}"))?,
    );
    wait_ready(metrics_port, &mut gw.0, &log)?;
    // The config holds the pool key; the gateway has read it.
    let _ = std::fs::remove_file(&cfg_path);
    Ok(Gateway {
        _nats: nats,
        _gw: gw,
        port,
        log,
    })
}

fn wait_ready(metrics_port: u16, gw: &mut Child, log: &Path) -> Result<(), Failed> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = gw.try_wait() {
            return Err(format!("gateway exited {status}: {}", tail(log)).into());
        }
        if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", metrics_port)) {
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

// --- Main ----------------------------------------------------------------------------------------

fn main() {
    let mut args = Arguments::from_args();
    let mut trials = Vec::new();
    let mut estimate_usd = 0.0;
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") && gateway_bin().exists() {
        let keys = Arc::new(env_keys());
        let seed = std::env::var("PARITY_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let cases = Arc::new(corpus(seed));
        for path in PATHS {
            if !keys.contains_key(path.model.provider.var) {
                continue;
            }
            for (i, case) in cases.iter().enumerate() {
                // Responses has no stop sequences.
                if case.kind == Kind::Stop
                    && (path.client == Dialect::Responses
                        || path.direct_dialect() == Dialect::Responses)
                {
                    continue;
                }
                let name = format!(
                    "{}::raw::{}::parity_{}",
                    claims_for(path, case),
                    path.route,
                    case.name
                );
                estimate_usd += estimate(path, case);
                let (path, keys, cases, n2) = (*path, keys.clone(), cases.clone(), name.clone());
                trials.push(Trial::test(name, move || {
                    run_trial(&n2, path, &cases[i], &keys)
                }));
            }
        }
        if !args.list {
            eprintln!(
                "parity: seed {seed}, {} cases, {} trials, estimated ${estimate_usd:.2} \
                 (up to 2x if every trial retries)",
                cases.len(),
                trials.len()
            );
        }
    }
    // Provider rate limits: a modest default concurrency.
    if args.test_threads.is_none() {
        args.test_threads = Some(6);
    }
    let list = args.list;
    let conclusion = libtest_mimic::run(&args, trials);
    gateways().lock().unwrap().clear();
    if !list && estimate_usd > 0.0 {
        eprintln!("parity: measured ${:.4}", *COST.lock().unwrap());
    }
    conclusion.exit();
}
