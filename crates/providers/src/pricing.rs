//! The reference pricer: one `ai.usage` row in; what we owe the vendor and what we charge the
//! customer out, in exact integer micro-dollars.
//!
//! This is the billing **contract**. The gateway calls [`price`] when it writes a row, and any other
//! implementation (a repricer, an invoice audit) must reproduce it exactly:
//! `verify/pricing_vectors.json` holds golden rows and their expected results, and
//! `crates/providers/ARCHITECTURE.md` ("Pricing contract") states every rule in prose. The rate
//! data lives in [`crate::rates`], generated from snapshots of each rate's primary source
//! (`verify/rates_sources/`) by `mise run rates:sync`.
//!
//! Pure: no I/O, no allocation, no floating point.
//!
//! # The model in one paragraph
//!
//! A row names a catalog model (`price_model`) and the provider that served it. The **price** is
//! the row's customer card ([`RowRates::customer`]: the primary vendor's published rates, whose
//! standard tier is exactly the catalog's [`crate::ListPrice`]) applied to the usage under that
//! vendor's own rules (long-context tier, fast mode, data residency, off-peak hours, server-tool
//! fees). The **cost** is the serving candidate's card applied the same way. On OpenRouter it is
//! the cost OpenRouter reported, plus its credit fee. Anything the tables cannot price exactly is
//! an [`Unpriced`] error, never a zero.

use crate::rates::{self, CostCard, OrEndpoint, RowRates};
use crate::{ProviderId, WireFormat};

/// USD per million tokens, as micro-dollars per million tokens: `"0.0375"` is `37_500`. Every
/// published rate has at most six decimal places per million tokens, so every rate is exact here.
pub type Rate = u64;

/// A dimensionless multiplier in basis points of one: `11_000` is ×1.1.
pub type Bps = u32;

/// One, in [`Bps`].
pub const ONE: Bps = 10_000;

/// Parse a USD decimal (`"12.5"`, `"0.0375"`) into a [`Rate`]. `const`, so the rate tables are
/// parsed by the compiler: a malformed literal (more than six decimals, a stray character) fails
/// the build rather than a request.
pub const fn usd(s: &str) -> Rate {
    let b = s.as_bytes();
    let mut i = 0;
    let mut whole: u64 = 0;
    let mut frac: u64 = 0;
    let mut frac_digits = 0;
    let mut seen_point = false;
    assert!(!b.is_empty(), "empty rate");
    while i < b.len() {
        let c = b[i];
        if c == b'.' {
            assert!(!seen_point, "two decimal points");
            seen_point = true;
        } else {
            assert!(c.is_ascii_digit(), "rate is not a decimal");
            let d = (c - b'0') as u64;
            if seen_point {
                frac = frac * 10 + d;
                frac_digits += 1;
                assert!(frac_digits <= 6, "more than six decimal places");
            } else {
                whole = whole * 10 + d;
            }
        }
        i += 1;
    }
    while frac_digits < 6 {
        frac *= 10;
        frac_digits += 1;
    }
    whole * 1_000_000 + frac
}

/// Per-token rates for one service class and context tier, in [`Rate`] units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenRates {
    /// Uncached input.
    pub input: Rate,
    /// Output (reasoning tokens are inside the output count).
    pub output: Rate,
    /// Cache reads. The input rate where the vendor publishes no discount.
    pub cache_read: Rate,
    /// Cache writes at the default (5-minute) TTL. The input rate where the vendor charges no
    /// write premium.
    pub cache_write_5m: Rate,
    /// Cache writes at the 1-hour TTL; `None` where the vendor sells none, so a row reporting
    /// them is [`Unpriced`] rather than billed at a guess.
    pub cache_write_1h: Option<Rate>,
}

impl TokenRates {
    /// `self × m` on every rate, rounded half-up at one micro-dollar per million tokens. Used
    /// only where the vendor states its tier as a multiple of the standard card (Anthropic's
    /// fast mode, xAI's long context and priority, DeepSeek's off-peak half); the truth test then
    /// holds every product to the vendor's own published figure where it publishes one.
    pub const fn times(self, m: Bps) -> TokenRates {
        const fn mul(r: Rate, m: Bps) -> Rate {
            (r * m as u64 + ONE as u64 / 2) / ONE as u64
        }
        TokenRates {
            input: mul(self.input, m),
            output: mul(self.output, m),
            cache_read: mul(self.cache_read, m),
            cache_write_5m: mul(self.cache_write_5m, m),
            cache_write_1h: match self.cache_write_1h {
                Some(r) => Some(mul(r, m)),
                None => None,
            },
        }
    }
}

/// The service class a request was **served** at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// The vendor's standard tier.
    Standard,
    /// Fast mode: Anthropic `speed: "fast"` (reported as `usage.speed`); OpenAI, xAI, Fireworks
    /// and OpenRouter `service_tier: "priority"` (OpenAI renamed Priority processing "Fast mode",
    /// and also echoes `"fast"`).
    Fast,
    /// OpenAI `service_tier: "ultrafast"`.
    Ultrafast,
    /// `service_tier: "flex"`.
    Flex,
    /// A time-of-day discount (DeepSeek's off-peak hours). Never read from the row: it is chosen
    /// from [`UsageRow::unix_secs`] for a standard-class request on a card that has one.
    OffPeak,
}

impl Class {
    pub const fn as_str(self) -> &'static str {
        match self {
            Class::Standard => "standard",
            Class::Fast => "fast",
            Class::Ultrafast => "ultrafast",
            Class::Flex => "flex",
            Class::OffPeak => "off_peak",
        }
    }
}

/// A long-context tier: the **whole request** is billed at the long rates when the prompt passes
/// the threshold. Every vendor that tiers counts the whole prompt (uncached input, cache reads and
/// cache writes) and re-prices every token of the request, not only those past the threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LongContext {
    pub threshold: u64,
    /// `true`: the tier applies at `prompt >= threshold` (xAI, "reaches"); `false`: at
    /// `prompt > threshold` (OpenAI, "more than 272K"; OpenRouter's `min_prompt_tokens`).
    pub inclusive: bool,
}

/// When a card's off-peak rates apply: every minute outside the peak windows, and the whole of
/// any holiday. Peak windows are minutes since 00:00 UTC, `[start, end)`, on weekdays only when
/// `weekdays_only`. A weekday peak minute on a day outside `[covered_from, covered_until]` (days
/// since 1970-01-01, UTC) is [`Unpriced::UnknownCalendar`]: the holiday list is published a year
/// at a time, and a missing year must refuse, not guess peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OffPeak {
    pub peak: &'static [(u16, u16)],
    pub weekdays_only: bool,
    /// Days (since the epoch, UTC) that are off-peak in full.
    pub holidays: &'static [u32],
    pub covered_from: u32,
    pub covered_until: u32,
    pub rates: TokenRates,
}

impl OffPeak {
    /// Whether `unix_secs` is off-peak; `None` when the calendar does not cover that day and the
    /// minute is a weekday peak minute.
    fn contains(&self, unix_secs: u64) -> Option<bool> {
        let day = unix_secs / 86_400;
        let minute = (unix_secs % 86_400) / 60;
        // 1970-01-01 was a Thursday: day 0 → weekday index 3 (Mon = 0).
        let weekday = (day + 3) % 7;
        if self.weekdays_only && weekday >= 5 {
            return Some(true);
        }
        let in_peak = self
            .peak
            .iter()
            .any(|&(s, e)| minute >= u64::from(s) && minute < u64::from(e));
        if !in_peak {
            return Some(true);
        }
        let day = u32::try_from(day).ok()?;
        if day < self.covered_from || day > self.covered_until {
            return None;
        }
        Some(self.holidays.contains(&day))
    }
}

/// A server-side tool kind, as the row's `server_tools` field names it
/// (`crates/gateway/ARCHITECTURE.md`, "Server-side tools"). A card prices a kind per call (or per
/// item); a kind it does not list is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tool {
    /// Web search: Anthropic `web_search_requests`, an OpenAI or xAI `web_search_call` search
    /// action, xAI `web_search_calls`, OpenRouter `web_search_requests`.
    WebSearch,
    /// An OpenAI `web_search_call` made with the `web_search_preview` tool (the row counts it apart
    /// when the request offered that tool): $10 / 1K on reasoning models, $25 / 1K on the others.
    WebSearchPreview,
    /// OpenAI `web_search_call` items with action `open_page` / `find_in_page`: no per-call fee.
    WebSearchPage,
    /// Anthropic `web_fetch_requests`: no per-call fee.
    WebFetch,
    /// Code execution calls: Anthropic `code_execution_requests` (billed per container-hour
    /// beyond a monthly free allowance, so no card prices it per row), OpenAI
    /// `code_interpreter_call` (a container session, billed per minute by memory size: never
    /// priced), xAI `code_interpreter_calls` ($5 / 1K).
    CodeExecution,
    /// OpenAI `file_search_call`, xAI `file_search_calls`.
    FileSearch,
    /// OpenAI `image_generation_call`, xAI `image_generation_calls`: billed at image rates the
    /// response does not report. Never priced.
    ImageGeneration,
    /// OpenAI `computer_call`: tokens only.
    ComputerUse,
    /// OpenAI `mcp_call`, xAI `mcp_calls`: tokens only.
    Mcp,
    /// OpenAI hosted `shell_call` / `local_shell_call`: a container session. Never priced.
    Shell,
    /// OpenAI `tool_search_call`: tokens only.
    ToolSearch,
    /// xAI `x_search_calls`: the call is free, the items it returns are billed
    /// ([`Self::XPosts`], [`Self::XUsers`]).
    XSearch,
    /// xAI `x_posts_fetched`.
    XPosts,
    /// xAI `x_users_fetched`.
    XUsers,
    /// xAI `document_search_calls`: no published fee under that name. Never priced.
    DocumentSearch,
    /// xAI `num_sources_used` (legacy Live Search). Never priced.
    Sources,
    /// OpenRouter `tool_calls_executed`: its server tools of every kind, web search included.
    /// Never added to the others: the pricer refuses a row whose `tool_calls` exceed its
    /// `web_search` (an OpenRouter tool it cannot price ran).
    ToolCalls,
}

impl Tool {
    pub const ALL: [Tool; 17] = [
        Tool::WebSearch,
        Tool::WebSearchPreview,
        Tool::WebSearchPage,
        Tool::WebFetch,
        Tool::CodeExecution,
        Tool::FileSearch,
        Tool::ImageGeneration,
        Tool::ComputerUse,
        Tool::Mcp,
        Tool::Shell,
        Tool::ToolSearch,
        Tool::XSearch,
        Tool::XPosts,
        Tool::XUsers,
        Tool::DocumentSearch,
        Tool::Sources,
        Tool::ToolCalls,
    ];
    pub const COUNT: usize = Self::ALL.len();

    /// The kind's name in the row's `server_tools` field.
    pub const fn as_str(self) -> &'static str {
        match self {
            Tool::WebSearch => "web_search",
            Tool::WebSearchPreview => "web_search_preview",
            Tool::WebSearchPage => "web_search_page",
            Tool::WebFetch => "web_fetch",
            Tool::CodeExecution => "code_execution",
            Tool::FileSearch => "file_search",
            Tool::ImageGeneration => "image_generation",
            Tool::ComputerUse => "computer_use",
            Tool::Mcp => "mcp",
            Tool::Shell => "shell",
            Tool::ToolSearch => "tool_search",
            Tool::XSearch => "x_search",
            Tool::XPosts => "x_posts",
            Tool::XUsers => "x_users",
            Tool::DocumentSearch => "document_search",
            Tool::Sources => "sources",
            Tool::ToolCalls => "tool_calls",
        }
    }

    pub fn parse(s: &str) -> Option<Tool> {
        Self::ALL.into_iter().find(|t| t.as_str() == s)
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Server-tool counts on one row, by kind. Fixed-size and `Copy`: no allocation per row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolCounts([u64; Tool::COUNT]);

impl ToolCounts {
    pub const fn new() -> Self {
        Self([0; Tool::COUNT])
    }
    pub const fn get(&self, t: Tool) -> u64 {
        self.0[t.index()]
    }
    pub fn set(&mut self, t: Tool, n: u64) {
        self.0[t.index()] = n;
    }
    pub const fn with(mut self, t: Tool, n: u64) -> Self {
        self.0[t.index()] = n;
        self
    }

    /// Read the row's `server_tools` text (`web_search=2,x_posts=14`). An unknown kind or a bad
    /// count is [`Unpriced::UnknownTool`]: a tool the pricer has never heard of may have a fee.
    pub fn parse(s: &str) -> Result<ToolCounts, Unpriced> {
        let mut out = ToolCounts::new();
        for pair in s.split(',').filter(|p| !p.is_empty()) {
            let (k, n) = pair.split_once('=').ok_or(Unpriced::UnknownTool)?;
            let t = Tool::parse(k).ok_or(Unpriced::UnknownTool)?;
            out.set(t, n.parse().map_err(|_| Unpriced::UnknownTool)?);
        }
        Ok(out)
    }

    fn used(&self) -> impl Iterator<Item = (Tool, u64)> + '_ {
        Tool::ALL
            .into_iter()
            .map(|t| (t, self.get(t)))
            .filter(|&(t, n)| n > 0 && t != Tool::ToolCalls)
    }
}

/// A full rate card: every published rate set, the rules that choose one, and per-call fees.
///
/// A class or tier that is `None` is not sold (or not published), and a row that needs it is
/// [`Unpriced::NoRate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card {
    pub standard: TokenRates,
    pub long: Option<TokenRates>,
    pub fast: Option<TokenRates>,
    pub fast_long: Option<TokenRates>,
    pub ultrafast: Option<TokenRates>,
    pub ultrafast_long: Option<TokenRates>,
    pub flex: Option<TokenRates>,
    pub flex_long: Option<TokenRates>,
    pub off_peak: Option<OffPeak>,
    pub long_context: Option<LongContext>,
    /// `inference_geo: "us"` (Anthropic data residency): a multiplier on every token rate.
    pub geo_us: Option<Bps>,
    /// Per-call fees, micro-dollars per call (or per item).
    pub tools: &'static [(Tool, u64)],
}

impl Card {
    /// A card with only a standard tier and no fees.
    pub const fn new(standard: TokenRates) -> Card {
        Card {
            standard,
            long: None,
            fast: None,
            fast_long: None,
            ultrafast: None,
            ultrafast_long: None,
            flex: None,
            flex_long: None,
            off_peak: None,
            long_context: None,
            geo_us: None,
            tools: &[],
        }
    }

    pub fn variant(&self, class: Class, long: bool) -> Option<&TokenRates> {
        match (class, long) {
            (Class::Standard, false) => Some(&self.standard),
            (Class::Standard, true) => self.long.as_ref(),
            (Class::Fast, false) => self.fast.as_ref(),
            (Class::Fast, true) => self.fast_long.as_ref(),
            (Class::Ultrafast, false) => self.ultrafast.as_ref(),
            (Class::Ultrafast, true) => self.ultrafast_long.as_ref(),
            (Class::Flex, false) => self.flex.as_ref(),
            (Class::Flex, true) => self.flex_long.as_ref(),
            (Class::OffPeak, false) => self.off_peak.as_ref().map(|o| &o.rates),
            (Class::OffPeak, true) => None,
        }
    }

    fn tool_fee(&self, t: Tool) -> Option<u64> {
        self.tools.iter().find(|(k, _)| *k == t).map(|&(_, f)| f)
    }
}

/// One `ai.usage` row, as the pricer reads it. Field names and meanings are the row's
/// (`crates/gateway/ARCHITECTURE.md`, "The `ai.usage` row").
///
/// `reasoning_tokens` is deliberately absent: reasoning is already inside `output_tokens` on every
/// wire the gateway meters (xAI's beside-count convention is folded in by the extractor), so it is
/// never priced on its own. The `requested_*` fields are absent too: the pricer prices what was
/// served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageRow<'a> {
    /// The catalog row to price at; `None` is unpriced.
    pub price_model: Option<&'a str>,
    /// The provider that served (`anthropic`, `bedrock`, `openrouter`, …); `None` when no
    /// provider was called.
    pub provider: Option<&'a str>,
    /// Which of the provider's prices applies to the endpoint: `regional` or `global` (Bedrock
    /// profiles; OpenRouter's in-region hosts). Every Bedrock candidate in the catalog is a `us.`
    /// profile (`regional`), and its card is the Regional SKUs; `global` there is refused. On
    /// OpenRouter `regional` is accepted: the reported cost carries the surcharge, and the
    /// fallback's dearest endpoint already includes the regional ones.
    pub price_variant: Option<&'a str>,
    /// Which convention `input_tokens` follows (see [`price`]).
    pub usage_wire: WireFormat,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// A subset of `cache_write_tokens`.
    pub cache_write_1h_tokens: u64,
    /// Cache writes from breakpoints the gateway added: already inside `input_tokens` (on both
    /// wires) and not in `cache_write_tokens`.
    pub gateway_cache_write_tokens: u64,
    /// The row's `server_tools`, parsed ([`ToolCounts::parse`]).
    pub server_tools: ToolCounts,
    /// The service tier the provider says it served at.
    pub service_tier: Option<&'a str>,
    /// The speed the provider says it served at (`standard` | `fast`).
    pub speed: Option<&'a str>,
    /// Where the provider says inference ran (`us` | `global`).
    pub inference_geo: Option<&'a str>,
    /// The vendor's own reported cost, a decimal USD string: OpenRouter `usage.cost`, xAI
    /// `cost_in_usd_ticks` ÷ 10^10.
    pub upstream_cost_usd: Option<&'a str>,
    /// OpenRouter's metered server-tool and plugin cost (`cost_details.server_tool_cost`).
    pub upstream_tool_cost_usd: Option<&'a str>,
    /// The host OpenRouter routed to (its provider name: `Anthropic`, `Amazon Bedrock`, …).
    pub served_by: Option<&'a str>,
    /// Some count on the row is the gateway's estimate.
    pub usage_estimated: bool,
    /// The stream ended early on a provider that keeps generating and billing: the row's tokens
    /// are a lower bound until reconciled.
    pub upstream_may_continue: bool,
    /// Served from the gateway's response cache: no vendor call was made.
    pub cache_hit: bool,
    /// When the request started, Unix seconds UTC (the row's timestamp less `latency_ms`).
    pub unix_secs: u64,
}

impl Default for UsageRow<'_> {
    fn default() -> Self {
        Self {
            price_model: None,
            provider: None,
            price_variant: None,
            usage_wire: WireFormat::Anthropic,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            cache_write_1h_tokens: 0,
            gateway_cache_write_tokens: 0,
            server_tools: ToolCounts::new(),
            service_tier: None,
            speed: None,
            inference_geo: None,
            upstream_cost_usd: None,
            upstream_tool_cost_usd: None,
            served_by: None,
            usage_estimated: false,
            upstream_may_continue: false,
            cache_hit: false,
            unix_secs: 0,
        }
    }
}

/// Whether the result is final.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Priced from the provider's reported usage (or a cache hit).
    Priced,
    /// Priced from the gateway's estimate of a cut-short stream ([`UsageRow::usage_estimated`]),
    /// or an OpenRouter cost with no reported cost, priced at the dearest matching endpoint.
    Estimated,
}

impl Status {
    pub const fn as_str(self) -> &'static str {
        match self {
            Status::Priced => "priced",
            Status::Estimated => "estimated",
        }
    }
}

/// Where the cost came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostBasis {
    /// The serving candidate's card times the row's tokens.
    Tokens,
    /// OpenRouter's reported `usage.cost`, plus its credit fee.
    Reported,
    /// OpenRouter with no reported cost: the dearest of its endpoints that could have served the
    /// row (the named host's, when the row names one), plus the credit fee. An upper bound.
    DearestEndpoint,
    /// A response-cache hit: no vendor call, so nothing owed.
    CacheHit,
}

impl CostBasis {
    pub const fn as_str(self) -> &'static str {
        match self {
            CostBasis::Tokens => "tokens",
            CostBasis::Reported => "reported",
            CostBasis::DearestEndpoint => "dearest_endpoint",
            CostBasis::CacheHit => "cache_hit",
        }
    }
}

/// Each component of one side, in micro-dollars, each rounded half-up on its own. The side's
/// total is rounded once from the exact sum, so the parts can differ from it by a few
/// micro-dollars. All zero on a reported cost, which OpenRouter does not break down.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Parts {
    pub input: u64,
    pub cache_read: u64,
    pub cache_write_5m: u64,
    pub cache_write_1h: u64,
    pub output: u64,
    pub tools: u64,
}

/// One side (cost or price) of a priced row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Side {
    pub micros: u64,
    /// The service class whose rates applied.
    pub class: Class,
    /// The long-context tier applied.
    pub long: bool,
    /// Product of every multiplier applied after the rates (data residency, OpenRouter's credit
    /// fee), in [`Bps`].
    pub multiplier: u64,
    pub parts: Parts,
}

/// The side's breakdown as one `key=value` list, written straight into the log line with no heap
/// allocation: `class=fast,long=false,mult=11000,input=24000,cache_read=36000,cache_write_5m=0,
/// cache_write_1h=0,output=100000,tools=30000` (micro-dollars; `mult` in basis points).
impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = &self.parts;
        write!(
            f,
            "class={},long={},mult={},input={},cache_read={},cache_write_5m={},cache_write_1h={},output={},tools={}",
            self.class.as_str(),
            self.long,
            self.multiplier,
            p.input,
            p.cache_read,
            p.cache_write_5m,
            p.cache_write_1h,
            p.output,
            p.tools
        )
    }
}

/// A priced row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Priced {
    pub status: Status,
    pub basis: CostBasis,
    /// What we owe the vendor.
    pub cost: Side,
    /// What we charge the customer.
    pub price: Side,
}

/// Why a row cannot be priced. Never a zero: the gateway logs the reason, and a human prices the
/// row (or fixes the table and reprices it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unpriced {
    /// The row names no catalog model.
    NoPriceModel,
    /// `price_model` is not a catalog row.
    UnknownModel,
    /// No provider was called, or the provider name is unknown.
    UnknownProvider,
    /// The serving provider is not a candidate of the row.
    NotACandidate,
    /// The candidate's rates could not be verified from a primary source.
    Unverified,
    /// `price_variant` names a variant the tables do not price for the serving provider.
    UnknownVariant,
    /// `service_tier` / `speed` name a class the tables do not price, or name two at once.
    UnknownClass,
    /// `inference_geo` names a residency the card does not price.
    UnknownGeo,
    /// The card sells no rate for the class and tier the row needs (fast mode on a model without
    /// it, 1-hour writes where none are sold, fast mode above 272K where unpublished).
    NoRate,
    /// A server tool was used that the card has no per-call fee for, or OpenRouter reports a
    /// metered tool or plugin cost (which no customer card prices).
    NoToolFee,
    /// `server_tools` names a kind the pricer does not know, or a count that is not a number.
    UnknownTool,
    /// Cache counts exceed the input they are part of (`openai` wire), the 1-hour writes exceed
    /// the writes, or the gateway's writes exceed the uncached input.
    InconsistentTokens,
    /// `upstream_cost_usd` or `upstream_tool_cost_usd` is not a non-negative decimal number.
    BadReportedCost,
    /// The OpenRouter host is not one the tables list for this model and class.
    UnknownHost,
    /// The request fell in a time-of-day peak window on a day the holiday calendar does not cover.
    UnknownCalendar,
    /// An intermediate exceeded 128 bits (a nonsense token count).
    Overflow,
}

impl Unpriced {
    /// Stable snake_case code for logs and vectors.
    pub const fn as_str(self) -> &'static str {
        match self {
            Unpriced::NoPriceModel => "no_price_model",
            Unpriced::UnknownModel => "unknown_model",
            Unpriced::UnknownProvider => "unknown_provider",
            Unpriced::NotACandidate => "not_a_candidate",
            Unpriced::Unverified => "unverified",
            Unpriced::UnknownVariant => "unknown_variant",
            Unpriced::UnknownClass => "unknown_class",
            Unpriced::UnknownGeo => "unknown_geo",
            Unpriced::NoRate => "no_rate",
            Unpriced::NoToolFee => "no_tool_fee",
            Unpriced::UnknownTool => "unknown_tool",
            Unpriced::InconsistentTokens => "inconsistent_tokens",
            Unpriced::BadReportedCost => "bad_reported_cost",
            Unpriced::UnknownHost => "unknown_host",
            Unpriced::UnknownCalendar => "unknown_calendar",
            Unpriced::Overflow => "overflow",
        }
    }
}

impl std::fmt::Display for Unpriced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for Unpriced {}

// ---------------------------------------------------------------------------------------------
// Exact arithmetic
// ---------------------------------------------------------------------------------------------
//
// Every component is an exact integer numerator over `DEN` micro-dollars: an amount in
// attodollars (1e-18 USD, 1e-12 µ$) times `SLOTS` multipliers in Bps, unused ones being `ONE`.
// A token component is `tokens × rate × 10^6` attodollars (a rate is µ$ per 10^6 tokens); a fee is
// `calls × µ$ × 10^12`. Each side is rounded once, half-up, at the end.

const ATTO_PER_MICRO: u128 = 1_000_000_000_000;
const SLOTS: u32 = 2;
const DEN: u128 = ATTO_PER_MICRO * (ONE as u128).pow(SLOTS);

/// The multipliers of one side, `SLOTS` of them. Two hold every combination the tables produce:
/// data residency (Anthropic) and the credit fee (OpenRouter) never meet on one card.
#[derive(Clone, Copy)]
struct Mults([Bps; SLOTS as usize]);

impl Mults {
    fn new(a: Bps, b: Bps) -> Mults {
        Mults([a, b])
    }
    fn factor(self) -> u128 {
        self.0.iter().map(|&m| u128::from(m)).product()
    }
    /// The combined multiplier in Bps, for the log. Exact for every pair the tables produce (one
    /// of the two is always `ONE`).
    fn bps(self) -> u64 {
        let one = u128::from(ONE).pow(SLOTS - 1);
        u64::try_from(self.factor() / one).unwrap_or(u64::MAX)
    }
}

fn round_half_up(num: u128) -> Result<u64, Unpriced> {
    let q = num.checked_add(DEN / 2).ok_or(Unpriced::Overflow)? / DEN;
    u64::try_from(q).map_err(|_| Unpriced::Overflow)
}

fn tokens_num(tokens: u64, rate: Rate, f: u128) -> Result<u128, Unpriced> {
    u128::from(tokens)
        .checked_mul(u128::from(rate))
        .and_then(|x| x.checked_mul(1_000_000))
        .and_then(|x| x.checked_mul(f))
        .ok_or(Unpriced::Overflow)
}

fn fee_num(calls: u64, micros: u64, f: u128) -> Result<u128, Unpriced> {
    u128::from(calls)
        .checked_mul(u128::from(micros))
        .and_then(|x| x.checked_mul(ATTO_PER_MICRO))
        .and_then(|x| x.checked_mul(f))
        .ok_or(Unpriced::Overflow)
}

/// Parse a JSON number (`0.00123`, `1.5e-05`, `12`) as exact attodollars, rounding half-up below
/// one attodollar (1e-18 USD). A negative, non-numeric or absurd value is refused.
pub fn parse_usd_atto(s: &str) -> Result<u128, Unpriced> {
    let bad = Unpriced::BadReportedCost;
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(i) => {
            let e = s.get(i + 1..).ok_or(bad)?;
            let e = e.strip_prefix('+').unwrap_or(e);
            (s.get(..i).ok_or(bad)?, e.parse::<i32>().map_err(|_| bad)?)
        }
        None => (s, 0),
    };
    let (whole, frac) = mant.split_once('.').unwrap_or((mant, ""));
    if (whole.is_empty() && frac.is_empty())
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || !frac.bytes().all(|b| b.is_ascii_digit())
        || !(-40..=40).contains(&exp)
    {
        return Err(bad);
    }
    let mut n: u128 = 0;
    let mut sig = 0usize;
    for d in whole.bytes().chain(frac.bytes()) {
        if n == 0 && d == b'0' {
            continue;
        }
        sig += 1;
        if sig > 36 {
            return Err(bad);
        }
        n = n * 10 + u128::from(d - b'0');
    }
    // The value is n × 10^(exp − frac.len()) USD = n × 10^(exp − frac.len() + 18) attodollars.
    let frac_len = i32::try_from(frac.len()).map_err(|_| bad)?;
    let shift = exp - frac_len + 18;
    if shift >= 0 {
        let p = 10u128
            .checked_pow(shift.unsigned_abs())
            .ok_or(Unpriced::Overflow)?;
        n.checked_mul(p).ok_or(Unpriced::Overflow)
    } else {
        // At most 36 significant digits, so n < 10^36: below 10^-36 of a unit it rounds to 0
        // (and 10^k past 38 would not fit in 128 bits).
        let k = shift.unsigned_abs();
        if k > 36 {
            return Ok(0);
        }
        let p = 10u128.pow(k);
        Ok((n + p / 2) / p)
    }
}

// ---------------------------------------------------------------------------------------------
// The pricer
// ---------------------------------------------------------------------------------------------

/// The row's tokens, normalized: every prompt token in exactly one bucket.
#[derive(Debug, Clone, Copy)]
struct Tokens {
    /// Uncached input, including the gateway's own cache writes.
    fresh: u64,
    /// The gateway's cache writes (a subset of `fresh`).
    gateway_writes: u64,
    read: u64,
    write_5m: u64,
    write_1h: u64,
    output: u64,
}

impl Tokens {
    fn of(r: &UsageRow<'_>) -> Result<Tokens, Unpriced> {
        let bad = Unpriced::InconsistentTokens;
        let write_5m = r
            .cache_write_tokens
            .checked_sub(r.cache_write_1h_tokens)
            .ok_or(bad)?;
        // OpenAI's convention counts cache reads (and OpenRouter's cache writes) inside
        // `input_tokens`; Anthropic's counts neither. Subtracting on the right wire is what keeps
        // a cached token from being billed twice, or not at all.
        let fresh = match r.usage_wire {
            WireFormat::Anthropic => r.input_tokens,
            WireFormat::OpenAi => r
                .input_tokens
                .checked_sub(r.cache_read_tokens)
                .and_then(|x| x.checked_sub(r.cache_write_tokens))
                .ok_or(bad)?,
        };
        if r.gateway_cache_write_tokens > fresh {
            return Err(bad);
        }
        Ok(Tokens {
            fresh,
            gateway_writes: r.gateway_cache_write_tokens,
            read: r.cache_read_tokens,
            write_5m,
            write_1h: r.cache_write_1h_tokens,
            output: r.output_tokens,
        })
    }

    /// The whole prompt: what every vendor's long-context threshold is measured on.
    fn prompt(&self) -> u64 {
        self.fresh
            .saturating_add(self.read)
            .saturating_add(self.write_5m)
            .saturating_add(self.write_1h)
    }
}

/// The class a row was served at, from its `service_tier` and `speed`.
fn served_class(r: &UsageRow<'_>) -> Result<Class, Unpriced> {
    let tier = match r.service_tier {
        // `default` (OpenAI), `standard` (Anthropic), `on_demand` (Groq): the standard tier.
        None | Some("default" | "standard" | "on_demand") => Class::Standard,
        Some("priority" | "fast") => Class::Fast,
        Some("ultrafast") => Class::Ultrafast,
        Some("flex") => Class::Flex,
        Some(_) => return Err(Unpriced::UnknownClass),
    };
    let speed = match r.speed {
        None | Some("standard") => Class::Standard,
        Some("fast") => Class::Fast,
        Some(_) => return Err(Unpriced::UnknownClass),
    };
    match (tier, speed) {
        (t, Class::Standard) => Ok(t),
        // OpenRouter reports an Anthropic fast request as `service_tier: "priority"` plus
        // `usage.speed: "fast"`: one class named twice.
        (Class::Standard | Class::Fast, Class::Fast) => Ok(Class::Fast),
        _ => Err(Unpriced::UnknownClass),
    }
}

fn geo_us(r: &UsageRow<'_>) -> Result<bool, Unpriced> {
    match r.inference_geo {
        // Anthropic reports `not_available` on models without data residency (the 4.5 family): no
        // premium, as global.
        None | Some("global" | "not_available") => Ok(false),
        Some("us") => Ok(true),
        Some(_) => Err(Unpriced::UnknownGeo),
    }
}

/// Price one side on one card. Returns the side and its exact numerator (for comparing sides).
fn side(
    card: &Card,
    t: &Tokens,
    r: &UsageRow<'_>,
    class: Class,
    geo_us: bool,
    fee: Bps,
) -> Result<(Side, u128), Unpriced> {
    let long = card.long_context.is_some_and(|lc| {
        if lc.inclusive {
            t.prompt() >= lc.threshold
        } else {
            t.prompt() > lc.threshold
        }
    });
    let class = match (class, &card.off_peak) {
        (Class::Standard, Some(op)) if !long => match op.contains(r.unix_secs) {
            Some(true) => Class::OffPeak,
            Some(false) => Class::Standard,
            None => return Err(Unpriced::UnknownCalendar),
        },
        _ => class,
    };
    let rates = card.variant(class, long).ok_or(Unpriced::NoRate)?;
    let geo = if geo_us {
        card.geo_us.ok_or(Unpriced::UnknownGeo)?
    } else {
        ONE
    };
    let m = Mults::new(geo, fee);
    let f = m.factor();

    // The gateway's own cache writes are inside the uncached input; the vendor bills them as
    // 5-minute writes, and pass-through bills them the same.
    let (input_tokens, gw_as_write) = (t.fresh - t.gateway_writes, t.gateway_writes);
    let write_1h_rate = if t.write_1h > 0 {
        rates.cache_write_1h.ok_or(Unpriced::NoRate)?
    } else {
        0
    };
    let input = tokens_num(input_tokens, rates.input, f)?;
    let read = tokens_num(t.read, rates.cache_read, f)?;
    let w5 = tokens_num(
        t.write_5m
            .checked_add(gw_as_write)
            .ok_or(Unpriced::Overflow)?,
        rates.cache_write_5m,
        f,
    )?;
    let w1 = tokens_num(t.write_1h, write_1h_rate, f)?;
    let output = tokens_num(t.output, rates.output, f)?;
    // Data residency multiplies "token pricing categories" only: a per-call fee carries the
    // credit fee (it was paid in OpenRouter credits) but not the geo premium.
    let tool_factor = Mults::new(ONE, fee).factor();
    let mut tools: u128 = 0;
    for (tool, n) in r.server_tools.used() {
        let per_call = card.tool_fee(tool).ok_or(Unpriced::NoToolFee)?;
        tools = tools
            .checked_add(fee_num(n, per_call, tool_factor)?)
            .ok_or(Unpriced::Overflow)?;
    }
    let total = [input, read, w5, w1, output, tools]
        .into_iter()
        .try_fold(0u128, u128::checked_add)
        .ok_or(Unpriced::Overflow)?;
    let parts = Parts {
        input: round_half_up(input)?,
        cache_read: round_half_up(read)?,
        cache_write_5m: round_half_up(w5)?,
        cache_write_1h: round_half_up(w1)?,
        output: round_half_up(output)?,
        tools: round_half_up(tools)?,
    };
    let side = Side {
        micros: round_half_up(total)?,
        class,
        long,
        multiplier: m.bps(),
        parts,
    };
    Ok((side, total))
}

/// Price one `ai.usage` row: what we owe the vendor (`cost`) and what we charge the customer
/// (`price`). See the module docs and `crates/providers/ARCHITECTURE.md` ("Pricing contract").
pub fn price(r: &UsageRow<'_>) -> Result<Priced, Unpriced> {
    // A response-cache hit made no vendor call: it costs nothing, and the customer is charged
    // what it cost us (owner decision, 2026-10-10). The row keeps its stored token facts for
    // audit.
    if r.cache_hit {
        let zero = Side {
            micros: 0,
            class: Class::Standard,
            long: false,
            multiplier: u64::from(ONE),
            parts: Parts::default(),
        };
        return Ok(Priced {
            status: Status::Priced,
            basis: CostBasis::CacheHit,
            cost: zero,
            price: zero,
        });
    }
    let model = r.price_model.ok_or(Unpriced::NoPriceModel)?;
    let row: &RowRates = rates::for_row(model).ok_or(Unpriced::UnknownModel)?;
    let provider = r
        .provider
        .and_then(crate::by_name)
        .ok_or(Unpriced::UnknownProvider)?
        .id;
    let cand = row
        .cost
        .iter()
        .find(|c| c.provider == provider)
        .ok_or(Unpriced::NotACandidate)?;
    match (provider, r.price_variant) {
        (_, None)
        | (ProviderId::Bedrock, Some("regional"))
        | (ProviderId::OpenRouter, Some("regional" | "global")) => {}
        _ => return Err(Unpriced::UnknownVariant),
    }
    let t = Tokens::of(r)?;
    let class = served_class(r)?;
    // Anthropic's own `service_tier: "priority"` is Priority Tier, committed capacity no longer
    // sold and priced by contract, not fast mode (which it reports as `usage.speed`). Only
    // OpenRouter spells fast mode `priority`.
    if matches!(provider, ProviderId::Anthropic | ProviderId::Bedrock)
        && r.service_tier.is_some_and(|t| t != "standard")
    {
        return Err(Unpriced::UnknownClass);
    }
    let geo = geo_us(r)?;

    let mut status = if r.usage_estimated || r.upstream_may_continue {
        Status::Estimated
    } else {
        Status::Priced
    };
    let (cost_side, basis) = match (&cand.card, r.upstream_cost_usd) {
        (CostCard::Unverified(_), _) => return Err(Unpriced::Unverified),
        // The vendor's own figure is the cost of goods wherever it reports one (OpenRouter, xAI).
        // It already carries the host's price, the tier, any regional premium and tool fees. On
        // OpenRouter it does not carry the fee on the credits that paid for it.
        (_, Some(raw)) => {
            let atto = parse_usd_atto(raw)?;
            let fee = if provider == ProviderId::OpenRouter {
                rates::OPENROUTER_CREDIT_FEE
            } else {
                ONE
            };
            let m = Mults::new(fee, ONE);
            let num = atto.checked_mul(m.factor()).ok_or(Unpriced::Overflow)?;
            let s = Side {
                micros: round_half_up(num)?,
                class,
                long: false,
                multiplier: m.bps(),
                parts: Parts::default(),
            };
            (s, CostBasis::Reported)
        }
        (CostCard::List, None) => {
            let (s, _) = side(&row.list, &t, r, class, geo, ONE)?;
            (s, CostBasis::Tokens)
        }
        (CostCard::Own(card), None) => {
            let (s, _) = side(card, &t, r, class, geo, ONE)?;
            (s, CostBasis::Tokens)
        }
        (CostCard::OpenRouter(endpoints), None) => {
            // OpenRouter's `tool_calls` counts its server tools of every kind, web search among
            // them. More of them than web searches means a tool ran whose fee no endpoint lists;
            // with no reported cost to carry it, refuse.
            if r.server_tools.get(Tool::ToolCalls) > r.server_tools.get(Tool::WebSearch) {
                return Err(Unpriced::NoToolFee);
            }
            status = Status::Estimated;
            (
                dearest_endpoint(endpoints, r, &t, class, geo)?,
                CostBasis::DearestEndpoint,
            )
        }
    };
    Ok(Priced {
        status,
        basis,
        cost: cost_side,
        // Pass-through (owner decision, 2026-10-10): the customer pays exactly what we pay for
        // this request, on the card of the host that served it, fees included.
        price: cost_side,
    })
}

/// OpenRouter by token math, when it reported no cost (a stream cut short before its usage
/// chunk): the dearest endpoint that could have served the row — those of the named host when
/// the row names one, all of the model's otherwise — among those listed for the served class.
/// OpenRouter lists each tier (`/fast`, `/flex`) and each region (`/us`, `/eu-west-1`) as its
/// own endpoint with its own price, and the row cannot say which served, so the cost is an upper
/// bound and the row is [`Status::Estimated`].
fn dearest_endpoint(
    endpoints: &[OrEndpoint],
    r: &UsageRow<'_>,
    t: &Tokens,
    class: Class,
    geo: bool,
) -> Result<Side, Unpriced> {
    if geo {
        // `inference_geo` is an Anthropic Messages field; OpenRouter routes by its own regional
        // endpoints, which are already in the list.
        return Err(Unpriced::UnknownGeo);
    }
    let mut best: Option<(Side, u128)> = None;
    for e in endpoints
        .iter()
        .filter(|e| e.class == class && r.served_by.is_none_or(|h| e.host.eq_ignore_ascii_case(h)))
    {
        let got = side(
            &e.card,
            t,
            r,
            Class::Standard,
            false,
            rates::OPENROUTER_CREDIT_FEE,
        )?;
        if best.as_ref().is_none_or(|b| got.1 > b.1) {
            best = Some(got);
        }
    }
    best.map(|(mut s, _)| {
        s.class = class;
        s
    })
    .ok_or(Unpriced::UnknownHost)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn usd_parses_exactly() {
        assert_eq!(usd("0.0375"), 37_500);
        assert_eq!(usd("12.5"), 12_500_000);
        assert_eq!(usd("0"), 0);
        assert_eq!(usd("0.000001"), 1);
    }

    #[test]
    fn reported_cost_parses_exactly() {
        let atto = parse_usd_atto;
        assert_eq!(atto("1"), Ok(1_000_000_000_000_000_000));
        assert_eq!(atto("0.0612345"), Ok(61_234_500_000_000_000));
        assert_eq!(atto("1.5e-05"), Ok(15_000_000_000_000));
        assert_eq!(atto("1.5E+2"), Ok(150_000_000_000_000_000_000));
        assert_eq!(atto(".5"), Ok(500_000_000_000_000_000));
        assert_eq!(atto("5."), Ok(5_000_000_000_000_000_000));
        // Below one attodollar: half-up.
        assert_eq!(atto("0.0000000000000000005"), Ok(1));
        assert_eq!(atto("0.0000000000000000004"), Ok(0));
        for bad in [
            "", ".", "-1", "+1", "1e", "abc", "1.2.3", " 1", "NaN", "inf", "1e999",
        ] {
            assert_eq!(atto(bad), Err(Unpriced::BadReportedCost), "{bad:?}");
        }
    }

    #[test]
    fn off_peak_reads_the_utc_weekday() {
        let rates = TokenRates {
            input: 2,
            output: 2,
            cache_read: 2,
            cache_write_5m: 2,
            cache_write_1h: None,
        };
        let op = OffPeak {
            peak: &[(60, 240)],
            weekdays_only: true,
            holidays: &[20_739],
            covered_from: 20_736,
            covered_until: 20_818,
            rates: rates.times(5_000),
        };
        assert_eq!(op.rates.input, 1);
        let at = |day: u64, min: u64| day * 86_400 + min * 60;
        // 2026-10-12 (day 20738) is a Monday; 20739 a recorded holiday; 20743 a Saturday.
        assert_eq!(op.contains(at(20_738, 61)), Some(false));
        assert_eq!(op.contains(at(20_738, 240)), Some(true));
        assert_eq!(op.contains(at(20_739, 61)), Some(true));
        assert_eq!(op.contains(at(20_743, 61)), Some(true));
        assert_eq!(op.contains(at(20_819, 61)), None);
        assert_eq!(op.contains(at(20_819, 300)), Some(true));
        // The covered span is inclusive at both ends.
        let every_day = OffPeak {
            weekdays_only: false,
            holidays: &[],
            covered_from: 20_738,
            covered_until: 20_740,
            ..op
        };
        assert_eq!(every_day.contains(at(20_737, 61)), None);
        assert_eq!(every_day.contains(at(20_738, 61)), Some(false));
        assert_eq!(every_day.contains(at(20_740, 61)), Some(false));
        assert_eq!(every_day.contains(at(20_741, 61)), None);
    }

    #[test]
    fn times_rounds_half_up_per_rate() {
        let r = TokenRates {
            input: 3,
            output: 1_000_000,
            cache_read: 1,
            cache_write_5m: 7,
            cache_write_1h: Some(5),
        };
        let x = r.times(15_000);
        assert_eq!(
            (
                x.input,
                x.output,
                x.cache_read,
                x.cache_write_5m,
                x.cache_write_1h
            ),
            (5, 1_500_000, 2, 11, Some(8))
        );
        assert_eq!(r.times(ONE), r);
        assert_eq!(r.times(25_000).output, 2_500_000);
    }

    #[test]
    fn reported_cost_digit_limits() {
        // Leading zeros are not significant digits.
        assert_eq!(
            parse_usd_atto("0000000000000000000000000000000000000001"),
            Ok(1_000_000_000_000_000_000)
        );
        // 36 significant digits are accepted, 37 refused.
        let d36 = format!("0.{}", "1".repeat(36));
        assert!(parse_usd_atto(&d36).is_ok());
        let d37 = format!("0.{}", "1".repeat(37));
        assert_eq!(parse_usd_atto(&d37), Err(Unpriced::BadReportedCost));
        // A value 10^-39 attodollars below a unit rounds to zero rather than overflowing.
        assert_eq!(parse_usd_atto("0.00000000000000001e-40"), Ok(0));
        // 36 nines × 10^-36 attodollars rounds up to one attodollar.
        let nines = format!("0.{}e-18", "9".repeat(36));
        assert_eq!(parse_usd_atto(&nines), Ok(1));
        assert_eq!(parse_usd_atto("123"), Ok(123_000_000_000_000_000_000));
    }

    #[test]
    fn server_tools_parse_the_row_text() {
        let t = ToolCounts::parse("web_search=2,x_posts=14").unwrap();
        assert_eq!((t.get(Tool::WebSearch), t.get(Tool::XPosts)), (2, 14));
        assert_eq!(ToolCounts::parse(""), Ok(ToolCounts::new()));
        for bad in ["web_search", "web_search=", "teleport=1", "web_search=-1"] {
            assert_eq!(ToolCounts::parse(bad), Err(Unpriced::UnknownTool), "{bad}");
        }
        for t in Tool::ALL {
            assert_eq!(Tool::parse(t.as_str()), Some(t));
        }
    }

    #[test]
    fn a_side_displays_its_breakdown() {
        let p = price(&UsageRow {
            inference_geo: Some("us"),
            server_tools: ToolCounts::new().with(Tool::WebSearch, 1),
            ..row("claude-opus-4-8", "anthropic")
        })
        .unwrap();
        assert_eq!(
            p.price.to_string(),
            "class=standard,long=false,mult=11000,input=5500,cache_read=0,cache_write_5m=0,\
             cache_write_1h=0,output=2750,tools=10000"
        );
    }

    #[test]
    fn unpriced_displays_its_code() {
        assert_eq!(Unpriced::NoToolFee.to_string(), "no_tool_fee");
    }

    fn row(model: &'static str, provider: &'static str) -> UsageRow<'static> {
        UsageRow {
            price_model: Some(model),
            provider: Some(provider),
            input_tokens: 1_000,
            output_tokens: 100,
            ..UsageRow::default()
        }
    }

    #[test]
    fn sides_report_their_multiplier() {
        let plain = price(&row("claude-opus-4-8", "anthropic")).unwrap();
        assert_eq!(plain.cost.multiplier, 10_000);
        let geo = price(&UsageRow {
            inference_geo: Some("us"),
            ..row("claude-opus-4-8", "anthropic")
        })
        .unwrap();
        assert_eq!(
            (geo.cost.multiplier, geo.price.multiplier),
            (11_000, 11_000)
        );
        let or = price(&UsageRow {
            usage_wire: WireFormat::OpenAi,
            upstream_cost_usd: Some("0.01"),
            ..row("claude-opus-4-8", "openrouter")
        })
        .unwrap();
        assert_eq!((or.cost.multiplier, or.price.multiplier), (10_550, 10_550));
    }

    #[test]
    fn gateway_writes_may_be_all_of_the_input() {
        let p = price(&UsageRow {
            gateway_cache_write_tokens: 1_000,
            ..row("claude-opus-4-8", "anthropic")
        })
        .unwrap();
        // 1000 gateway writes at $6.25 (what the vendor bills) + 100 output at $25, passed through.
        assert_eq!((p.price.micros, p.cost.micros), (8_750, 8_750));
        assert_eq!(p.cost.parts.cache_write_5m, 6_250);
        assert_eq!(p.price.parts.input, 0);
    }

    /// A long request is billed at the long rates even inside an off-peak window: off-peak
    /// rates are a short-context tier.
    #[test]
    fn long_context_wins_over_off_peak() {
        let rates = |r| TokenRates {
            input: r,
            output: r,
            cache_read: r,
            cache_write_5m: r,
            cache_write_1h: None,
        };
        let card = Card {
            long: Some(rates(3)),
            long_context: Some(LongContext {
                threshold: 10,
                inclusive: false,
            }),
            off_peak: Some(OffPeak {
                peak: &[],
                weekdays_only: false,
                holidays: &[],
                covered_from: 0,
                covered_until: 0,
                rates: rates(1),
            }),
            ..Card::new(rates(2))
        };
        let r = UsageRow {
            input_tokens: 1_000_000,
            ..UsageRow::default()
        };
        let t = Tokens::of(&r).unwrap();
        let (s, _) = side(&card, &t, &r, Class::Standard, false, ONE).unwrap();
        assert_eq!((s.class, s.long, s.micros), (Class::Standard, true, 3));
        let short = UsageRow {
            input_tokens: 10,
            ..UsageRow::default()
        };
        let t = Tokens::of(&short).unwrap();
        let (s, _) = side(&card, &t, &short, Class::Standard, false, ONE).unwrap();
        assert_eq!((s.class, s.long), (Class::OffPeak, false));
    }

    /// The pricer runs once per `ai.usage` row, after the response. Measured, not assumed: run
    /// with `cargo test --release -p beyond-ai-providers --lib -- --ignored price_cost --nocapture`.
    #[test]
    #[ignore = "timing probe"]
    fn price_cost() {
        let rows = [
            UsageRow {
                price_model: Some("claude-opus-5-5"),
                provider: Some("anthropic"),
                input_tokens: 3_000,
                cache_read_tokens: 90_000,
                cache_write_tokens: 20_000,
                cache_write_1h_tokens: 5_000,
                output_tokens: 2_500,
                speed: Some("fast"),
                inference_geo: Some("us"),
                server_tools: ToolCounts::new().with(Tool::WebSearch, 3),
                ..UsageRow::default()
            },
            UsageRow {
                price_model: Some("z-ai/glm-5.3"),
                provider: Some("openrouter"),
                usage_wire: WireFormat::OpenAi,
                input_tokens: 30_000,
                cache_read_tokens: 10_000,
                output_tokens: 2_000,
                ..UsageRow::default()
            },
        ];
        for r in &rows {
            let n = 1_000_000u32;
            let start = std::time::Instant::now();
            for _ in 0..n {
                std::hint::black_box(price(std::hint::black_box(r))).unwrap();
            }
            println!(
                "{:?}: {:.0} ns per row",
                r.price_model,
                start.elapsed().as_nanos() as f64 / f64::from(n)
            );
        }
    }
}
