//! The reference pricer: one `ai.usage` row in; what we owe the vendor and what we charge the
//! customer out, in exact integer micro-dollars.
//!
//! This is the billing **contract**. The gateway calls [`price`] when it writes a row, and any other
//! implementation (a repricer, an invoice audit) must reproduce it exactly:
//! `verify/pricing_vectors.json` holds golden rows and their expected results, and
//! `crates/providers/ARCHITECTURE.md` ("Pricing contract") states every rule in prose. The rate
//! data lives in [`crate::rates`]; every rate there carries its primary source in
//! `verify/catalog_truth.toml`, and a test holds the two to each other.
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

/// A server-side tool a vendor runs and bills per call (or per item), beside the tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tool {
    /// Web search: Anthropic `server_tool_use.web_search_requests`, an OpenAI `web_search_call`
    /// search action, xAI `web_search_calls`, OpenRouter native search.
    WebSearch,
    /// OpenAI's `web_search_preview` tool (priced apart on non-reasoning models).
    WebSearchPreview,
    /// Anthropic web fetch (`server_tool_use.web_fetch_requests`): tokens only, a $0 fee.
    WebFetch,
    /// One X post returned by xAI's `x_search` (`x_posts_fetched`).
    XSearchPost,
    /// One X profile returned by xAI's `x_search` (`x_users_fetched`).
    XSearchProfile,
    /// Code execution: xAI `code_interpreter_calls` (per call); Anthropic
    /// `server_tool_use.code_execution_requests` (billed per container-hour beyond a monthly free
    /// allowance, so no card prices it per row).
    CodeExecution,
    /// xAI attachment search.
    AttachmentSearch,
    /// File / collections search: OpenAI `file_search_call`, xAI `file_search_calls`.
    FileSearch,
    /// An OpenAI hosted container session (Code Interpreter, hosted Shell). Billed per minute of
    /// session at a memory-size rate, neither of which the response reports: never priced.
    ContainerSession,
    /// An image-generation tool call (OpenAI `image_generation_call`, xAI
    /// `image_generation_calls`). Billed at image-token rates the response does not report:
    /// never priced.
    ImageGeneration,
    /// One request of OpenRouter's web plugin (`plugins: [{id: "web"}]`, `:online`). Its price
    /// depends on the search engine and result count, which the row does not carry: never
    /// priced from tokens (OpenRouter's reported cost carries it).
    OpenRouterWeb,
}

impl Tool {
    pub const ALL: [Tool; 11] = [
        Tool::WebSearch,
        Tool::WebSearchPreview,
        Tool::WebFetch,
        Tool::XSearchPost,
        Tool::XSearchProfile,
        Tool::CodeExecution,
        Tool::AttachmentSearch,
        Tool::FileSearch,
        Tool::ContainerSession,
        Tool::ImageGeneration,
        Tool::OpenRouterWeb,
    ];
    pub const COUNT: usize = Self::ALL.len();

    /// The stable snake_case name the vectors (and the row's per-kind counts) use.
    pub const fn as_str(self) -> &'static str {
        match self {
            Tool::WebSearch => "web_search",
            Tool::WebSearchPreview => "web_search_preview",
            Tool::WebFetch => "web_fetch",
            Tool::XSearchPost => "x_search_post",
            Tool::XSearchProfile => "x_search_profile",
            Tool::CodeExecution => "code_execution",
            Tool::AttachmentSearch => "attachment_search",
            Tool::FileSearch => "file_search",
            Tool::ContainerSession => "container_session",
            Tool::ImageGeneration => "image_generation",
            Tool::OpenRouterWeb => "openrouter_web",
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
    fn used(&self) -> impl Iterator<Item = (Tool, u64)> + '_ {
        Tool::ALL
            .into_iter()
            .map(|t| (t, self.get(t)))
            .filter(|&(_, n)| n > 0)
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

/// One `ai.usage` row, as the pricer reads it. Field names are the row's.
///
/// `reasoning_tokens` is deliberately absent: reasoning is already inside `output_tokens` on every
/// wire the gateway meters (xAI's beside-count convention is folded in by the extractor), so it is
/// never priced on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageRow<'a> {
    /// The catalog row to price at; `None` (a model the catalog does not carry) is unpriced.
    pub price_model: Option<&'a str>,
    /// The provider that served (`anthropic`, `bedrock`, `openrouter`, …); `None` when no
    /// provider was called.
    pub provider: Option<&'a str>,
    /// A routing variant of the serving candidate that changes what we pay. Only `us` on Bedrock
    /// (whose catalog candidates are all `us.` geo profiles already) is accepted; anything else is
    /// [`Unpriced::UnknownVariant`]. OpenRouter's regional endpoints need no variant: the reported
    /// cost carries the premium, and without it the dearest endpoint is used.
    pub upstream_variant: Option<&'a str>,
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
    pub server_tools: ToolCounts,
    /// The service tier the provider says it served at.
    pub service_tier: Option<&'a str>,
    /// The speed the provider says it served at (Anthropic `usage.speed`: `standard` | `fast`).
    pub speed: Option<&'a str>,
    /// Where the provider says inference ran (Anthropic `usage.inference_geo`: `us` | `global`).
    pub inference_geo: Option<&'a str>,
    /// OpenRouter's reported `usage.cost`, as the raw JSON number text (USD).
    pub openrouter_cost: Option<&'a str>,
    /// The host OpenRouter served from (its provider name: `Anthropic`, `Amazon Bedrock`, …).
    pub openrouter_host: Option<&'a str>,
    /// The token counts are the gateway's estimate of a cut-short stream.
    pub usage_estimated: bool,
    /// Served from the gateway's response cache: no vendor call was made.
    pub cache_hit: bool,
    /// When the request started, Unix seconds UTC (DeepSeek's off-peak hours).
    pub unix_secs: u64,
}

impl Default for UsageRow<'_> {
    fn default() -> Self {
        Self {
            price_model: None,
            provider: None,
            upstream_variant: None,
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
            openrouter_cost: None,
            openrouter_host: None,
            usage_estimated: false,
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
    /// `upstream_variant` names a variant the tables do not price.
    UnknownVariant,
    /// `service_tier` / `speed` name a class the tables do not price, or name two at once.
    UnknownClass,
    /// `inference_geo` names a residency the card does not price.
    UnknownGeo,
    /// The card sells no rate for the class and tier the row needs (fast mode on a model without
    /// it, 1-hour writes where none are sold, fast mode above 272K where unpublished).
    NoRate,
    /// A server tool was used that the card has no per-call fee for.
    NoToolFee,
    /// Cache counts exceed the input they are part of (`openai` wire), the 1-hour writes exceed
    /// the writes, or the gateway's writes exceed the uncached input.
    InconsistentTokens,
    /// `openrouter_cost` is not a non-negative decimal number.
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
    fn bps(self) -> u64 {
        let one = u128::from(ONE).pow(SLOTS - 1);
        u64::try_from((self.factor() + one / 2) / one).unwrap_or(u64::MAX)
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
        let k = shift.unsigned_abs();
        if k > 38 {
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
        None | Some("global") => Ok(false),
        Some("us") => Ok(true),
        Some(_) => Err(Unpriced::UnknownGeo),
    }
}

/// Which rate a side bills the gateway's own cache writes at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GatewayWrites {
    /// The customer: the input rate (the gateway chose to cache, not the client).
    AsInput,
    /// The vendor: the 5-minute write rate it actually charges.
    AsWrites,
}

/// Price one side on one card. Returns the side and its exact numerator (for comparing sides).
fn side(
    card: &Card,
    t: &Tokens,
    r: &UsageRow<'_>,
    class: Class,
    geo_us: bool,
    fee: Bps,
    gw: GatewayWrites,
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

    let (input_tokens, gw_as_write) = match gw {
        GatewayWrites::AsInput => (t.fresh, 0),
        GatewayWrites::AsWrites => (t.fresh - t.gateway_writes, t.gateway_writes),
    };
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
    match (provider, r.upstream_variant) {
        (_, None) | (ProviderId::Bedrock, Some("us")) => {}
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

    let (price_side, _) = side(
        &row.customer,
        &t,
        r,
        class,
        geo,
        ONE,
        GatewayWrites::AsInput,
    )?;

    let mut status = if r.usage_estimated {
        Status::Estimated
    } else {
        Status::Priced
    };
    let (cost_side, basis) = match &cand.card {
        CostCard::Unverified(_) => return Err(Unpriced::Unverified),
        CostCard::Customer => {
            let (s, _) = side(
                &row.customer,
                &t,
                r,
                class,
                geo,
                ONE,
                GatewayWrites::AsWrites,
            )?;
            (s, CostBasis::Tokens)
        }
        CostCard::Own(card) => {
            let (s, _) = side(card, &t, r, class, geo, ONE, GatewayWrites::AsWrites)?;
            (s, CostBasis::Tokens)
        }
        CostCard::OpenRouter(endpoints) => {
            if let Some(raw) = r.openrouter_cost {
                // The reported cost already carries the host's price, the tier, any regional
                // premium and plugin fees. What it does not carry is the fee on the credits that
                // paid for it.
                let atto = parse_usd_atto(raw)?;
                let m = Mults::new(rates::OPENROUTER_CREDIT_FEE, ONE);
                let num = atto.checked_mul(m.factor()).ok_or(Unpriced::Overflow)?;
                let s = Side {
                    micros: round_half_up(num)?,
                    class,
                    long: false,
                    multiplier: m.bps(),
                    parts: Parts::default(),
                };
                (s, CostBasis::Reported)
            } else {
                status = Status::Estimated;
                (
                    dearest_endpoint(endpoints, r, &t, class, geo)?,
                    CostBasis::DearestEndpoint,
                )
            }
        }
    };
    Ok(Priced {
        status,
        basis,
        cost: cost_side,
        price: price_side,
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
    for e in endpoints.iter().filter(|e| {
        e.class == class
            && r.openrouter_host
                .is_none_or(|h| e.host.eq_ignore_ascii_case(h))
    }) {
        let got = side(
            &e.card,
            t,
            r,
            Class::Standard,
            false,
            rates::OPENROUTER_CREDIT_FEE,
            GatewayWrites::AsWrites,
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
