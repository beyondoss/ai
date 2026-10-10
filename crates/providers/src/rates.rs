//! The rate tables behind [`crate::pricing`]: per catalog row, what the customer pays (the
//! primary vendor's card) and what each candidate costs us.
//!
//! Every rate here is a vendor's published figure, read from a primary source on the date in
//! `verify/catalog_truth.toml` (`[[card]]`, `[[pricing]]`, `[[openrouter]]`), and
//! `rates_match_truth` holds this file to that one, entry for entry. A customer card's standard
//! tier is the catalog's [`crate::ListPrice`] for the row (`customer_standard_is_the_list_price`),
//! so `GET /v1/models` and the invoice can never disagree.
//!
//! Changing any rate changes [`RATE_VERSION`], which every priced row logs: a row can always be
//! traced to the exact table that priced it, and repriced against another.
//!
//! # Maintenance
//!
//! A vendor reprices: update the card here and its `[[card]]` entry (rates, `source`, `date`),
//! then `rate_version_names_this_table` prints the new [`RATE_VERSION`]. OpenRouter's endpoint
//! lists change often; refresh them from `GET /api/v1/models/{slug}/endpoints` (prices there are
//! USD per token: multiply by 10^6) and keep the `[[openrouter]]` entries in step.

use crate::ProviderId;
use crate::pricing::{Bps, Card, Class, LongContext, OffPeak, TokenRates, Tool, usd};

/// The exact rate table compiled into this build: the date its rates were checked, and a hash of
/// the table (`rate_version_names_this_table` recomputes it, and fails with the new value when
/// any rate changes). Logged on every priced `ai.usage` row.
pub const RATE_VERSION: &str = "2026-10-10.a06079be9e5b5e77";

/// OpenRouter's fee on the credits that pay for every request: 5.5% of each card purchase on the
/// Standard plan ("OpenRouter's fee is charged when you buy credits, 5.5% on Standard and 8% on
/// Business, never on individual requests", <https://openrouter.ai/pricing>, 2026-10-10; the $0.80
/// minimum per purchase is a purchase-level floor, not a per-request one). Applied on top of
/// `usage.cost`, which is inference at the provider's list price. A Business plan is 8%: change
/// this, and the truth entry, if the account moves.
pub const OPENROUTER_CREDIT_FEE: Bps = 10_550;

/// What one catalog row costs and charges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRates {
    /// The catalog row's name.
    pub model: &'static str,
    /// What the customer pays: the primary vendor's card. Its standard tier is the row's
    /// [`crate::ListPrice`].
    pub customer: Card,
    /// What each candidate costs us, one per provider in the row's `candidates` and `responses`.
    pub cost: &'static [CandidateCost],
}

/// What one candidate costs us.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateCost {
    pub provider: ProviderId,
    pub card: CostCard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostCard {
    /// The customer card: the candidate is the vendor whose card the customer pays.
    Customer,
    /// The candidate's own published card.
    Own(&'static Card),
    /// OpenRouter: its reported `usage.cost` plus [`OPENROUTER_CREDIT_FEE`], or, with none
    /// reported, the dearest of these endpoints.
    OpenRouter(&'static [OrEndpoint]),
    /// No primary source could be found for this candidate's rates: every row it serves is
    /// unpriced. The reason is the truth file's.
    Unverified(&'static str),
}

/// One OpenRouter endpoint for a model: a host, a region or tier variant, and its prices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrEndpoint {
    /// OpenRouter's provider name (`Anthropic`, `Amazon Bedrock`, `Google`, …).
    pub host: &'static str,
    /// OpenRouter's endpoint tag (`amazon-bedrock/us`, `openai/flex`, …).
    pub tag: &'static str,
    /// The service class this endpoint serves (its tag's `/fast`, `/priority`, `/flex`,
    /// `/ultrafast` suffix; standard otherwise).
    pub class: Class,
    /// Its prices. A time-of-day schedule is folded to its dearest window.
    pub card: Card,
}

/// The rates for a catalog row, by exact name.
pub fn for_row(model: &str) -> Option<&'static RowRates> {
    ROW_RATES
        .binary_search_by(|r| r.model.cmp(model))
        .ok()
        .and_then(|i| ROW_RATES.get(i))
}

/// Token rates from USD-per-million decimal strings.
const fn tr(
    input: &str,
    output: &str,
    cache_read: &str,
    cache_write_5m: &str,
    cache_write_1h: Option<&str>,
) -> TokenRates {
    TokenRates {
        input: usd(input),
        output: usd(output),
        cache_read: usd(cache_read),
        cache_write_5m: usd(cache_write_5m),
        cache_write_1h: match cache_write_1h {
            Some(s) => Some(usd(s)),
            None => None,
        },
    }
}

// --- per-call fee lists ---
const ANTHROPIC_TOOLS: &[(Tool, u64)] = &[(Tool::WebFetch, 0), (Tool::WebSearch, 10000)];
const OPENAI_NON_REASONING_TOOLS: &[(Tool, u64)] = &[
    (Tool::ComputerUse, 0),
    (Tool::FileSearch, 2500),
    (Tool::Mcp, 0),
    (Tool::ToolSearch, 0),
    (Tool::WebSearchPage, 0),
];
const OPENAI_TOOLS: &[(Tool, u64)] = &[
    (Tool::ComputerUse, 0),
    (Tool::FileSearch, 2500),
    (Tool::Mcp, 0),
    (Tool::ToolSearch, 0),
    (Tool::WebSearch, 10000),
    (Tool::WebSearchPage, 0),
];
const XAI_TOOLS: &[(Tool, u64)] = &[
    (Tool::CodeExecution, 5000),
    (Tool::FileSearch, 2500),
    (Tool::Mcp, 0),
    (Tool::WebSearch, 5000),
    (Tool::XPosts, 5000),
    (Tool::XSearch, 0),
    (Tool::XUsers, 10000),
];
const OR_SEARCH_10: &[(Tool, u64)] = &[(Tool::WebSearch, 10000)];
const OR_SEARCH_5: &[(Tool, u64)] = &[(Tool::WebSearch, 5000)];

// --- vendor cards ---
/// `anthropic:claude-fable-5`.
const ANTHROPIC_CLAUDE_FABLE_5: Card = Card {
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("10", "50", "1", "12.5", Some("20")))
};

/// `anthropic:claude-fable-5-1`.
const ANTHROPIC_CLAUDE_FABLE_5_1: Card = Card {
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("10", "50", "0.25", "12.5", Some("20")))
};

/// `anthropic:claude-haiku-4-5`.
const ANTHROPIC_CLAUDE_HAIKU_4_5: Card = Card {
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("1", "5", "0.1", "1.25", Some("2")))
};

/// `anthropic:claude-opus-4-5`.
const ANTHROPIC_CLAUDE_OPUS_4_5: Card = Card {
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
};

/// `anthropic:claude-opus-4-6`.
const ANTHROPIC_CLAUDE_OPUS_4_6: Card = Card {
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
};

/// `anthropic:claude-opus-4-7`.
const ANTHROPIC_CLAUDE_OPUS_4_7: Card = Card {
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
};

/// `anthropic:claude-opus-4-8`.
const ANTHROPIC_CLAUDE_OPUS_4_8: Card = Card {
    fast: Some(tr("10", "50", "1", "12.5", Some("20"))),
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
};

/// `anthropic:claude-opus-5`.
const ANTHROPIC_CLAUDE_OPUS_5: Card = Card {
    fast: Some(tr("10", "50", "1", "12.5", Some("20"))),
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
};

/// `anthropic:claude-opus-5-5`.
const ANTHROPIC_CLAUDE_OPUS_5_5: Card = Card {
    fast: Some(tr("8", "40", "0.4", "10", Some("16"))),
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("4", "20", "0.2", "5", Some("8")))
};

/// `anthropic:claude-sonnet-4-5`.
const ANTHROPIC_CLAUDE_SONNET_4_5: Card = Card {
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
};

/// `anthropic:claude-sonnet-4-6`.
const ANTHROPIC_CLAUDE_SONNET_4_6: Card = Card {
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
};

/// `anthropic:claude-sonnet-5`.
const ANTHROPIC_CLAUDE_SONNET_5: Card = Card {
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("2", "10", "0.2", "2.5", Some("4")))
};

/// `anthropic:claude-sonnet-5-5`.
const ANTHROPIC_CLAUDE_SONNET_5_5: Card = Card {
    geo_us: Some(11000),
    tools: ANTHROPIC_TOOLS,
    ..Card::new(tr("2", "10", "0.1", "2.5", Some("4")))
};

/// `bedrock:claude-haiku-4-5`.
const BEDROCK_CLAUDE_HAIKU_4_5: Card = Card::new(tr("1.1", "5.5", "0.11", "1.375", Some("2.2")));

/// `bedrock:claude-opus-4-8`.
const BEDROCK_CLAUDE_OPUS_4_8: Card = Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")));

/// `deepseek:deepseek-flash`.
const DEEPSEEK_DEEPSEEK_FLASH: Card = Card {
    off_peak: Some(OffPeak {
        peak: &[(60, 240), (360, 600)],
        weekdays_only: true,
        holidays: &[],
        covered_from: 20736,
        covered_until: 20818,
        rates: tr("0.15", "0.6", "0.003", "0.15", None),
    }),
    ..Card::new(tr("0.3", "1.2", "0.006", "0.3", None))
};

/// `deepseek:deepseek-v4-pro`.
const DEEPSEEK_DEEPSEEK_V4_PRO: Card = Card {
    off_peak: Some(OffPeak {
        peak: &[(60, 240), (360, 600)],
        weekdays_only: true,
        holidays: &[],
        covered_from: 20736,
        covered_until: 20818,
        rates: tr("0.66", "1.98", "0.022", "0.66", None),
    }),
    ..Card::new(tr("1.32", "3.96", "0.044", "1.32", None))
};

/// `together:Qwen/Qwen3.5-9B`.
const TOGETHER_QWEN_QWEN3_5_9B: Card = Card::new(tr("0.17", "0.25", "0.17", "0.17", None));

/// `together:Qwen/Qwen3.6-Plus`.
const TOGETHER_QWEN_QWEN3_6_PLUS: Card = Card::new(tr("0.5", "3", "0.5", "0.5", None));

/// `together:Qwen/Qwen3.7-Max`.
const TOGETHER_QWEN_QWEN3_7_MAX: Card = Card::new(tr("2.5", "7.5", "0.5", "2.5", None));

/// `together:Qwen/Qwen3.7-Plus`.
const TOGETHER_QWEN_QWEN3_7_PLUS: Card = Card::new(tr("0.32", "1.28", "0.32", "0.32", None));

/// `together:Qwen/Qwen3.8-2.4T-A95B`.
const TOGETHER_QWEN_QWEN3_8_2_4T_A95B: Card = Card::new(tr("2", "6", "0.5", "2", None));

/// `together:Qwen/Qwen3.8-Flash`.
const TOGETHER_QWEN_QWEN3_8_FLASH: Card = Card::new(tr("0.15", "0.47", "0.15", "0.15", None));

/// `together:meta-llama/Llama-3.3-70B-Instruct-Turbo`.
const TOGETHER_META_LLAMA_LLAMA_3_3_70B_INSTRUCT_TURBO: Card =
    Card::new(tr("1.04", "1.04", "1.04", "1.04", None));

/// `together:meta-models/Muse-Glimmer-30B`.
const TOGETHER_META_MODELS_MUSE_GLIMMER_30B: Card =
    Card::new(tr("0.35", "1.5", "0.04", "0.35", None));

/// `together:MiniMaxAI/MiniMax-M3`.
const TOGETHER_MINIMAXAI_MINIMAX_M3: Card = Card::new(tr("0.3", "1.2", "0.06", "0.3", None));

/// `together:moonshotai/Kimi-K3`.
const TOGETHER_MOONSHOTAI_KIMI_K3: Card = Card::new(tr("3", "15", "0.3", "3", None));

/// `together:thinkingmachines/Inkling`.
const TOGETHER_THINKINGMACHINES_INKLING: Card = Card::new(tr("1", "4.05", "0.17", "1", None));

/// `together:zai-org/GLM-5.2`.
const TOGETHER_ZAI_ORG_GLM_5_2: Card = Card::new(tr("1.4", "4.4", "0.26", "1.4", None));

/// `together:zai-org/GLM-5.3`.
const TOGETHER_ZAI_ORG_GLM_5_3: Card = Card::new(tr("1.4", "4.4", "0.26", "1.4", None));

/// `together:zai-org/GLM-5.3-Flash`.
const TOGETHER_ZAI_ORG_GLM_5_3_FLASH: Card = Card::new(tr("0.15", "0.5", "0.03", "0.15", None));

/// `together:deepseek-ai/DeepSeek-V4-Pro-0813`.
const TOGETHER_DEEPSEEK_AI_DEEPSEEK_V4_PRO_0813: Card =
    Card::new(tr("1.32", "3.96", "0.13", "1.32", None));

/// `together:deepseek-ai/DeepSeek-V4.1-Flash`.
const TOGETHER_DEEPSEEK_AI_DEEPSEEK_V4_1_FLASH: Card =
    Card::new(tr("0.3", "1.2", "0.006", "0.3", None));

/// `together:openai/gpt-oss-120b`.
const TOGETHER_OPENAI_GPT_OSS_120B: Card = Card::new(tr("0.15", "0.6", "0.15", "0.15", None));

/// `fireworks:accounts/fireworks/models/gpt-oss-120b`.
const FIREWORKS_ACCOUNTS_FIREWORKS_MODELS_GPT_OSS_120B: Card = Card {
    fast: Some(tr("0.18", "0.72", "0.018", "0.18", None)),
    ..Card::new(tr("0.15", "0.6", "0.015", "0.15", None))
};

/// `fireworks:accounts/fireworks/models/kimi-k3`.
const FIREWORKS_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3: Card = Card {
    fast: Some(tr("3.75", "18.75", "0.375", "3.75", None)),
    ..Card::new(tr("3", "15", "0.3", "3", None))
};

/// `fireworks:accounts/fireworks/models/minimax-m3`.
const FIREWORKS_ACCOUNTS_FIREWORKS_MODELS_MINIMAX_M3: Card = Card {
    fast: Some(tr("0.45", "1.8", "0.09", "0.45", None)),
    ..Card::new(tr("0.3", "1.2", "0.06", "0.3", None))
};

/// `groq:openai/gpt-oss-120b`.
const GROQ_OPENAI_GPT_OSS_120B: Card = Card {
    flex: Some(tr("0.15", "0.6", "0.075", "0.15", None)),
    ..Card::new(tr("0.15", "0.6", "0.075", "0.15", None))
};

/// `groq:openai/gpt-oss-20b`.
const GROQ_OPENAI_GPT_OSS_20B: Card = Card {
    flex: Some(tr("0.075", "0.3", "0.0375", "0.075", None)),
    ..Card::new(tr("0.075", "0.3", "0.0375", "0.075", None))
};

/// `groq:openai/gpt-oss-safeguard-20b`.
const GROQ_OPENAI_GPT_OSS_SAFEGUARD_20B: Card = Card {
    flex: Some(tr("0.075", "0.3", "0.0375", "0.075", None)),
    ..Card::new(tr("0.075", "0.3", "0.0375", "0.075", None))
};

/// `groq:qwen/qwen3.8-27b`.
const GROQ_QWEN_QWEN3_8_27B: Card = Card {
    flex: Some(tr("0.8", "4", "0.8", "0.8", None)),
    ..Card::new(tr("0.8", "4", "0.8", "0.8", None))
};

/// `openai:gpt-4.1`.
const OPENAI_GPT_4_1: Card = Card {
    fast: Some(tr("3.5", "14", "0.875", "3.5", None)),
    tools: OPENAI_NON_REASONING_TOOLS,
    ..Card::new(tr("2", "8", "0.5", "2", None))
};

/// `openai:gpt-4.1-mini`.
const OPENAI_GPT_4_1_MINI: Card = Card {
    fast: Some(tr("0.7", "2.8", "0.175", "0.7", None)),
    tools: OPENAI_NON_REASONING_TOOLS,
    ..Card::new(tr("0.4", "1.6", "0.1", "0.4", None))
};

/// `openai:gpt-4o`.
const OPENAI_GPT_4O: Card = Card {
    fast: Some(tr("4.25", "17", "2.125", "4.25", None)),
    tools: OPENAI_NON_REASONING_TOOLS,
    ..Card::new(tr("2.5", "10", "1.25", "2.5", None))
};

/// `openai:gpt-4o-mini`.
const OPENAI_GPT_4O_MINI: Card = Card {
    fast: Some(tr("0.25", "1", "0.125", "0.25", None)),
    tools: OPENAI_NON_REASONING_TOOLS,
    ..Card::new(tr("0.15", "0.6", "0.075", "0.15", None))
};

/// `openai:gpt-5`.
const OPENAI_GPT_5: Card = Card {
    fast: Some(tr("2.5", "20", "0.25", "2.5", None)),
    flex: Some(tr("0.625", "5", "0.0625", "0.625", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("1.25", "10", "0.125", "1.25", None))
};

/// `openai:gpt-5-mini`.
const OPENAI_GPT_5_MINI: Card = Card {
    fast: Some(tr("0.45", "3.6", "0.045", "0.45", None)),
    flex: Some(tr("0.125", "1", "0.0125", "0.125", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("0.25", "2", "0.025", "0.25", None))
};

/// `openai:gpt-5-nano`.
const OPENAI_GPT_5_NANO: Card = Card {
    flex: Some(tr("0.025", "0.2", "0.0025", "0.025", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("0.05", "0.4", "0.005", "0.05", None))
};

/// `openai:gpt-5-pro`.
const OPENAI_GPT_5_PRO: Card = Card {
    tools: OPENAI_TOOLS,
    ..Card::new(tr("15", "120", "15", "15", None))
};

/// `openai:gpt-5.1`.
const OPENAI_GPT_5_1: Card = Card {
    fast: Some(tr("2.5", "20", "0.25", "2.5", None)),
    flex: Some(tr("0.625", "5", "0.0625", "0.625", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("1.25", "10", "0.125", "1.25", None))
};

/// `openai:gpt-5.2`.
const OPENAI_GPT_5_2: Card = Card {
    fast: Some(tr("3.5", "28", "0.35", "3.5", None)),
    flex: Some(tr("0.875", "7", "0.0875", "0.875", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("1.75", "14", "0.175", "1.75", None))
};

/// `openai:gpt-5.2-pro`.
const OPENAI_GPT_5_2_PRO: Card = Card {
    tools: OPENAI_TOOLS,
    ..Card::new(tr("21", "168", "21", "21", None))
};

/// `openai:gpt-5.3-codex`.
const OPENAI_GPT_5_3_CODEX: Card = Card {
    fast: Some(tr("3.5", "28", "0.35", "3.5", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("1.75", "14", "0.175", "1.75", None))
};

/// `openai:gpt-5.4`.
const OPENAI_GPT_5_4: Card = Card {
    long: Some(tr("5", "22.5", "0.5", "5", None)),
    fast: Some(tr("5", "30", "0.5", "5", None)),
    flex: Some(tr("1.25", "7.5", "0.13", "1.25", None)),
    flex_long: Some(tr("2.5", "11.25", "0.25", "2.5", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("2.5", "15", "0.25", "2.5", None))
};

/// `openai:gpt-5.4-mini`.
const OPENAI_GPT_5_4_MINI: Card = Card {
    fast: Some(tr("1.5", "9", "0.15", "1.5", None)),
    flex: Some(tr("0.375", "2.25", "0.0375", "0.375", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("0.75", "4.5", "0.075", "0.75", None))
};

/// `openai:gpt-5.4-nano`.
const OPENAI_GPT_5_4_NANO: Card = Card {
    flex: Some(tr("0.1", "0.625", "0.01", "0.1", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("0.2", "1.25", "0.02", "0.2", None))
};

/// `openai:gpt-5.4-pro`.
const OPENAI_GPT_5_4_PRO: Card = Card {
    long: Some(tr("60", "270", "60", "60", None)),
    flex: Some(tr("15", "90", "15", "15", None)),
    flex_long: Some(tr("30", "135", "30", "30", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("30", "180", "30", "30", None))
};

/// `openai:gpt-5.5`.
const OPENAI_GPT_5_5: Card = Card {
    long: Some(tr("10", "45", "1", "10", None)),
    fast: Some(tr("12.5", "75", "1.25", "12.5", None)),
    flex: Some(tr("2.5", "15", "0.25", "2.5", None)),
    flex_long: Some(tr("5", "22.5", "0.5", "5", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("5", "30", "0.5", "5", None))
};

/// `openai:gpt-5.5-pro`.
const OPENAI_GPT_5_5_PRO: Card = Card {
    long: Some(tr("60", "270", "60", "60", None)),
    flex: Some(tr("15", "90", "15", "15", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("30", "180", "30", "30", None))
};

/// `openai:gpt-5.6-luna`.
const OPENAI_GPT_5_6_LUNA: Card = Card {
    long: Some(tr("0.4", "1.8", "0.04", "0.5", None)),
    fast: Some(tr("0.4", "2.4", "0.04", "0.5", None)),
    fast_long: Some(tr("0.8", "3.6", "0.08", "1", None)),
    flex: Some(tr("0.1", "0.6", "0.01", "0.125", None)),
    flex_long: Some(tr("0.2", "0.9", "0.02", "0.25", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("0.2", "1.2", "0.02", "0.25", None))
};

/// `openai:gpt-5.6-sol`.
const OPENAI_GPT_5_6_SOL: Card = Card {
    long: Some(tr("8", "30", "0.8", "10", None)),
    fast: Some(tr("8", "40", "0.8", "10", None)),
    fast_long: Some(tr("16", "60", "1.6", "20", None)),
    flex: Some(tr("2", "10", "0.2", "2.5", None)),
    flex_long: Some(tr("4", "15", "0.4", "5", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("4", "20", "0.4", "5", None))
};

/// `openai:gpt-5.6-terra`.
const OPENAI_GPT_5_6_TERRA: Card = Card {
    long: Some(tr("4", "18", "0.4", "5", None)),
    fast: Some(tr("4", "24", "0.4", "5", None)),
    fast_long: Some(tr("8", "36", "0.8", "10", None)),
    flex: Some(tr("1", "6", "0.1", "1.25", None)),
    flex_long: Some(tr("2", "9", "0.2", "2.5", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("2", "12", "0.2", "2.5", None))
};

/// `openai:gpt-6-astra`.
const OPENAI_GPT_6_ASTRA: Card = Card {
    long: Some(tr("20", "75", "2", "25", None)),
    fast: Some(tr("20", "100", "2", "25", None)),
    fast_long: Some(tr("40", "150", "4", "50", None)),
    ultrafast: Some(tr("60", "300", "6", "75", None)),
    ultrafast_long: Some(tr("120", "450", "12", "150", None)),
    flex: Some(tr("5", "25", "0.5", "6.25", None)),
    flex_long: Some(tr("10", "37.5", "1", "12.5", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("10", "50", "1", "12.5", None))
};

/// `openai:gpt-6-luna`.
const OPENAI_GPT_6_LUNA: Card = Card {
    long: Some(tr("0.2", "0.75", "0.02", "0.25", None)),
    fast: Some(tr("0.2", "1", "0.02", "0.25", None)),
    fast_long: Some(tr("0.4", "1.5", "0.04", "0.5", None)),
    flex: Some(tr("0.05", "0.25", "0.005", "0.0625", None)),
    flex_long: Some(tr("0.1", "0.375", "0.01", "0.125", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("0.1", "0.5", "0.01", "0.125", None))
};

/// `openai:gpt-6-sol`.
const OPENAI_GPT_6_SOL: Card = Card {
    long: Some(tr("4", "15", "0.4", "5", None)),
    fast: Some(tr("4", "20", "0.4", "5", None)),
    fast_long: Some(tr("8", "30", "0.8", "10", None)),
    flex: Some(tr("1", "5", "0.1", "1.25", None)),
    flex_long: Some(tr("2", "7.5", "0.2", "2.5", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("2", "10", "0.2", "2.5", None))
};

/// `openai:gpt-6.1-sol`.
const OPENAI_GPT_6_1_SOL: Card = Card {
    long: Some(tr("4", "15", "0.2", "5", None)),
    fast: Some(tr("4", "20", "0.2", "5", None)),
    fast_long: Some(tr("8", "30", "0.4", "10", None)),
    ultrafast: Some(tr("12", "60", "0.6", "15", None)),
    ultrafast_long: Some(tr("24", "90", "1.2", "30", None)),
    flex: Some(tr("1", "5", "0.05", "1.25", None)),
    flex_long: Some(tr("2", "7.5", "0.1", "2.5", None)),
    long_context: Some(LongContext {
        threshold: 272000,
        inclusive: false,
    }),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("2", "10", "0.1", "2.5", None))
};

/// `xai:grok-4.20`.
const XAI_GROK_4_20: Card = Card {
    long: Some(tr("2.5", "5", "0.4", "2.5", None)),
    fast: Some(tr("2.5", "5", "0.4", "2.5", None)),
    long_context: Some(LongContext {
        threshold: 200000,
        inclusive: true,
    }),
    tools: XAI_TOOLS,
    ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
};

/// `xai:grok-4.20-multi-agent`.
const XAI_GROK_4_20_MULTI_AGENT: Card = Card {
    long: Some(tr("2.5", "5", "0.4", "2.5", None)),
    fast: Some(tr("2.5", "5", "0.4", "2.5", None)),
    long_context: Some(LongContext {
        threshold: 200000,
        inclusive: true,
    }),
    tools: XAI_TOOLS,
    ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
};

/// `xai:grok-4.3`.
const XAI_GROK_4_3: Card = Card {
    long: Some(tr("2.5", "5", "0.4", "2.5", None)),
    fast: Some(tr("2.5", "5", "0.4", "2.5", None)),
    long_context: Some(LongContext {
        threshold: 200000,
        inclusive: true,
    }),
    tools: XAI_TOOLS,
    ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
};

/// `xai:grok-4.5`.
const XAI_GROK_4_5: Card = Card {
    long: Some(tr("4", "12", "0.6", "4", None)),
    fast: Some(tr("4", "12", "0.6", "4", None)),
    long_context: Some(LongContext {
        threshold: 200000,
        inclusive: true,
    }),
    tools: XAI_TOOLS,
    ..Card::new(tr("2", "6", "0.3", "2", None))
};

/// `xai:grok-4.6`.
const XAI_GROK_4_6: Card = Card {
    long: Some(tr("4", "12", "1", "4", None)),
    fast: Some(tr("4", "12", "1", "4", None)),
    long_context: Some(LongContext {
        threshold: 200000,
        inclusive: true,
    }),
    tools: XAI_TOOLS,
    ..Card::new(tr("2", "6", "0.5", "2", None))
};

/// `xai:grok-4.7`.
const XAI_GROK_4_7: Card = Card {
    long: Some(tr("4", "12", "1", "4", None)),
    fast: Some(tr("4", "12", "1", "4", None)),
    long_context: Some(LongContext {
        threshold: 200000,
        inclusive: true,
    }),
    tools: XAI_TOOLS,
    ..Card::new(tr("2", "6", "0.5", "2", None))
};

/// `xai:grok-build-0.1`.
const XAI_GROK_BUILD_0_1: Card = Card {
    long: Some(tr("2", "4", "0.4", "2", None)),
    fast: Some(tr("2", "4", "0.4", "2", None)),
    long_context: Some(LongContext {
        threshold: 200000,
        inclusive: true,
    }),
    tools: XAI_TOOLS,
    ..Card::new(tr("1", "2", "0.2", "1", None))
};

/// `openai:o3`.
const OPENAI_O3: Card = Card {
    fast: Some(tr("3.5", "14", "0.875", "3.5", None)),
    flex: Some(tr("1", "4", "0.25", "1", None)),
    tools: OPENAI_TOOLS,
    ..Card::new(tr("2", "8", "0.5", "2", None))
};

/// `openai:o3-pro`.
const OPENAI_O3_PRO: Card = Card {
    tools: OPENAI_TOOLS,
    ..Card::new(tr("20", "80", "20", "20", None))
};

/// `openai:text-embedding-3-large`.
const OPENAI_TEXT_EMBEDDING_3_LARGE: Card = Card::new(tr("0.13", "0", "0.13", "0.13", None));

/// `openai:text-embedding-3-small`.
const OPENAI_TEXT_EMBEDDING_3_SMALL: Card = Card::new(tr("0.02", "0", "0.02", "0.02", None));

// --- OpenRouter endpoints ---
/// `anthropic/claude-fable-5` (`GET /api/v1/models/anthropic/claude-fable-5/endpoints`).
static OR_ANTHROPIC_CLAUDE_FABLE_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", Some("20")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", Some("20")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", Some("20")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", Some("20")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", Some("20")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("11", "55", "1.1", "13.75", Some("22")))
        },
    },
];

/// `anthropic/claude-fable-5.1` (`GET /api/v1/models/anthropic/claude-fable-5.1/endpoints`).
static OR_ANTHROPIC_CLAUDE_FABLE_5_1: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "0.25", "12.5", Some("20")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "0.25", "12.5", Some("20")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "0.25", "12.5", Some("20")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "0.25", "12.5", Some("20")))
        },
    },
];

/// `anthropic/claude-haiku-4.5` (`GET /api/v1/models/anthropic/claude-haiku-4.5/endpoints`).
static OR_ANTHROPIC_CLAUDE_HAIKU_4_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.1", "1.25", Some("2")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.1", "1.25", Some("2")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.1", "1.25", Some("2")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.1", "1.25", Some("2")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.1", "5.5", "0.11", "1.375", Some("2.2")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us-east5",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.1", "5.5", "0.11", "1.375", Some("2.2")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.1", "5.5", "0.11", "1.375", Some("2.2")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.1", "5.5", "0.11", "1.375", Some("2.2")))
        },
    },
];

/// `anthropic/claude-opus-4.5` (`GET /api/v1/models/anthropic/claude-opus-4.5/endpoints`).
static OR_ANTHROPIC_CLAUDE_OPUS_4_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
];

/// `anthropic/claude-opus-4.6` (`GET /api/v1/models/anthropic/claude-opus-4.6/endpoints`).
static OR_ANTHROPIC_CLAUDE_OPUS_4_6: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
];

/// `anthropic/claude-opus-4.7` (`GET /api/v1/models/anthropic/claude-opus-4.7/endpoints`).
static OR_ANTHROPIC_CLAUDE_OPUS_4_7: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
];

/// `anthropic/claude-opus-4.8` (`GET /api/v1/models/anthropic/claude-opus-4.8/endpoints`).
static OR_ANTHROPIC_CLAUDE_OPUS_4_8: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic/fast",
        class: Class::Fast,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", Some("20")))
        },
    },
];

/// `anthropic/claude-opus-5` (`GET /api/v1/models/anthropic/claude-opus-5/endpoints`).
static OR_ANTHROPIC_CLAUDE_OPUS_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", Some("10")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "27.5", "0.55", "6.875", Some("11")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic/fast",
        class: Class::Fast,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", Some("20")))
        },
    },
];

/// `anthropic/claude-opus-5.5` (`GET /api/v1/models/anthropic/claude-opus-5.5/endpoints`).
static OR_ANTHROPIC_CLAUDE_OPUS_5_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.2", "5", Some("8")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.2", "5", Some("8")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.2", "5", Some("8")))
        },
    },
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.2", "5", Some("8")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.2", "5", Some("8")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.22", "5.5", Some("8.8")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.22", "5.5", Some("8.8")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.22", "5.5", Some("8.8")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.22", "5.5", Some("8.8")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.22", "5.5", Some("8.8")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic/fast",
        class: Class::Fast,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("8", "40", "0.4", "10", Some("16")))
        },
    },
];

/// `anthropic/claude-sonnet-4.5` (`GET /api/v1/models/anthropic/claude-sonnet-4.5/endpoints`).
static OR_ANTHROPIC_CLAUDE_SONNET_4_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            long: Some(tr("6", "22.5", "0.6", "7.5", Some("12"))),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            long: Some(tr("6", "22.5", "0.6", "7.5", Some("12"))),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            long: Some(tr("6", "22.5", "0.6", "7.5", Some("12"))),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            long: Some(tr("6", "22.5", "0.6", "7.5", Some("12"))),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("6.6", "24.75", "0.66", "8.25", Some("13.2"))),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("3.3", "16.5", "0.33", "4.125", Some("6.6")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us-east5",
        class: Class::Standard,
        card: Card {
            long: Some(tr("6.6", "24.75", "0.66", "8.25", Some("13.2"))),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("3.3", "16.5", "0.33", "4.125", Some("6.6")))
        },
    },
];

/// `anthropic/claude-sonnet-4.6` (`GET /api/v1/models/anthropic/claude-sonnet-4.6/endpoints`).
static OR_ANTHROPIC_CLAUDE_SONNET_4_6: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3", "15", "0.3", "3.75", Some("6")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3.3", "16.5", "0.33", "4.125", Some("6.6")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3.3", "16.5", "0.33", "4.125", Some("6.6")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3.3", "16.5", "0.33", "4.125", Some("6.6")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us-east5",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3.3", "16.5", "0.33", "4.125", Some("6.6")))
        },
    },
];

/// `anthropic/claude-sonnet-5` (`GET /api/v1/models/anthropic/claude-sonnet-5/endpoints`).
static OR_ANTHROPIC_CLAUDE_SONNET_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", Some("4.4")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", Some("4.4")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", Some("4.4")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", Some("4.4")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", Some("4.4")))
        },
    },
];

/// `anthropic/claude-sonnet-5.5` (`GET /api/v1/models/anthropic/claude-sonnet-5.5/endpoints`).
static OR_ANTHROPIC_CLAUDE_SONNET_5_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/global",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Claude Platform on AWS",
        tag: "claude-on-aws",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Anthropic",
        tag: "anthropic",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", Some("4")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.11", "2.75", Some("4.4")))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.11", "2.75", Some("4.4")))
        },
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/europe",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.11", "2.75", Some("4.4")))
        },
    },
];

/// `deepseek/deepseek-v4.1-flash` (`GET /api/v1/models/deepseek/deepseek-v4.1-flash/endpoints`).
static OR_DEEPSEEK_DEEPSEEK_V4_1_FLASH: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Relace",
        tag: "relace",
        class: Class::Standard,
        card: Card::new(tr("0.0001", "0.6", "0.005", "0.0001", None)),
    },
    OrEndpoint {
        host: "OpenInference",
        tag: "open-inference/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.00011", "0.36", "0.00011", "0.00011", None)),
    },
    OrEndpoint {
        host: "Morph",
        tag: "morph/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.029", "1", "0.009", "0.029", None)),
    },
    OrEndpoint {
        host: "Wafer",
        tag: "wafer",
        class: Class::Standard,
        card: Card::new(tr("0.05", "1.6", "0.049", "0.05", None)),
    },
    OrEndpoint {
        host: "InferenceNet",
        tag: "inference-net/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.07", "0.6", "0.01", "0.07", None)),
    },
    OrEndpoint {
        host: "Sail Research",
        tag: "sail-research/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.08", "0.4", "0.01", "0.08", None)),
    },
    OrEndpoint {
        host: "Decart",
        tag: "decart/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.09", "0.18", "0.018", "0.09", None)),
    },
    OrEndpoint {
        host: "Ionstream",
        tag: "ionstream",
        class: Class::Standard,
        card: Card::new(tr("0.1", "1.1", "0.01", "0.1", None)),
    },
    OrEndpoint {
        host: "DekaLLM",
        tag: "dekallm",
        class: Class::Standard,
        card: Card::new(tr("0.12", "1.2", "0.005", "0.12", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.14", "0.42", "0.0042", "0.14", None)),
    },
    OrEndpoint {
        host: "StreamLake",
        tag: "streamlake/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.147", "0.588", "0.00294", "0.147", None)),
    },
    OrEndpoint {
        host: "Baidu",
        tag: "baidu/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.1497", "0.5988", "0.002994", "0.1497", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.03", "0.3", None)),
    },
    OrEndpoint {
        host: "DeepSeek",
        tag: "deepseek",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.006", "0.3", None)),
    },
    OrEndpoint {
        host: "DigitalOcean",
        tag: "digitalocean",
        class: Class::Standard,
        card: Card::new(tr("0.165", "0.66", "0.006", "0.165", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.18", "0.72", "0.0036", "0.18", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.195", "0.78", "0.0039", "0.195", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.2", "0.65", "0.03", "0.2", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala",
        class: Class::Standard,
        card: Card::new(tr("0.21", "0.84", "0.0042", "0.21", None)),
    },
    OrEndpoint {
        host: "Makora",
        tag: "makora/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.27", "1.15", "0.009", "0.27", None)),
    },
    OrEndpoint {
        host: "Crusoe",
        tag: "crusoe/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.29", "1.2", "0.007", "0.29", None)),
    },
    OrEndpoint {
        host: "AtlasCloud",
        tag: "atlas-cloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.03", "0.3", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.007", "0.3", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.006", "0.3", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.006", "0.3", None)),
    },
    OrEndpoint {
        host: "Modal",
        tag: "modal",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.03", "0.3", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.007", "0.3", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.006", "0.3", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.006", "0.3", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.0075", "0.3", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks/us",
        class: Class::Standard,
        card: Card::new(tr("0.45", "1.8", "0.009", "0.45", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fast",
        class: Class::Fast,
        card: Card::new(tr("0.6", "2.4", "0.014", "0.6", None)),
    },
];

/// `deepseek/deepseek-v4-pro-0813` (`GET /api/v1/models/deepseek/deepseek-v4-pro-0813/endpoints`).
static OR_DEEPSEEK_DEEPSEEK_V4_PRO_0813: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Baidu",
        tag: "baidu/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.21912", "0.65736", "0.006972", "0.21912", None)),
    },
    OrEndpoint {
        host: "Wafer",
        tag: "wafer",
        class: Class::Standard,
        card: Card::new(tr("0.3", "5", "0.219", "0.3", None)),
    },
    OrEndpoint {
        host: "Ionstream",
        tag: "ionstream",
        class: Class::Standard,
        card: Card::new(tr("0.37", "2.93", "0.088", "0.37", None)),
    },
    OrEndpoint {
        host: "Sail Research",
        tag: "sail-research/us",
        class: Class::Standard,
        card: Card::new(tr("0.4", "3", "0.033", "0.4", None)),
    },
    OrEndpoint {
        host: "Sail Research",
        tag: "sail-research/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.4", "3", "0.033", "0.4", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card::new(tr("1.122", "3.366", "0.1122", "1.122", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/us-east-1",
        class: Class::Standard,
        card: Card::new(tr("1.32", "3.96", "0.132", "1.32", None)),
    },
    OrEndpoint {
        host: "StreamLake",
        tag: "streamlake",
        class: Class::Standard,
        card: Card::new(tr("0.66", "1.98", "0.022", "0.66", None)),
    },
    OrEndpoint {
        host: "DeepSeek",
        tag: "deepseek",
        class: Class::Standard,
        card: Card::new(tr("1.32", "3.96", "0.044", "1.32", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala",
        class: Class::Standard,
        card: Card::new(tr("0.957", "2.8776", "0.099", "0.957", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.99", "2.97", "0.033", "0.99", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.056", "3.168", "0.0352", "1.056", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.3", "2.6", "0.1", "1.3", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.31", "3.96", "0.044", "1.31", None)),
    },
    OrEndpoint {
        host: "AtlasCloud",
        tag: "atlas-cloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.32", "3.96", "0.132", "1.32", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.32", "3.96", "0.044", "1.32", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("1.32", "3.96", "0.13", "1.32", None)),
    },
    OrEndpoint {
        host: "DigitalOcean",
        tag: "digitalocean",
        class: Class::Standard,
        card: Card::new(tr("1.32", "3.96", "0.044", "1.32", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.32", "3.96", "0.044", "1.32", None)),
    },
    OrEndpoint {
        host: "Cloudflare",
        tag: "cloudflare",
        class: Class::Standard,
        card: Card::new(tr("1.32", "3.96", "0.044", "1.32", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice",
        class: Class::Standard,
        card: Card::new(tr("1.65", "4.95", "0.165", "1.65", None)),
    },
];

/// `google/gemma-4-31b-it` (`GET /api/v1/models/google/gemma-4-31b-it/endpoints`).
static OR_GOOGLE_GEMMA_4_31B_IT: &[OrEndpoint] = &[
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/turbo",
        class: Class::Standard,
        card: Card::new(tr("0.09", "0.34", "0.05", "0.09", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.1", "0.34", "0.1", "0.1", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.12", "0.36", "0.09", "0.12", None)),
    },
    OrEndpoint {
        host: "Chutes",
        tag: "chutes/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.12", "0.37", "0.012", "0.12", None)),
    },
    OrEndpoint {
        host: "Crusoe",
        tag: "crusoe/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.14", "0.4", "0.14", "0.14", None)),
    },
    OrEndpoint {
        host: "Friendli",
        tag: "friendli",
        class: Class::Standard,
        card: Card::new(tr("0.14", "0.4", "0.14", "0.14", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.14", "0.4", "0.14", "0.14", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.4", "0.06", "0.15", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.2", "0.4", "0.2", "0.2", None)),
    },
    OrEndpoint {
        host: "Io Net",
        tag: "io-net",
        class: Class::Standard,
        card: Card::new(tr("0.361", "1.0925", "0.1805", "0.361", None)),
    },
    OrEndpoint {
        host: "SambaNova",
        tag: "sambanova",
        class: Class::Standard,
        card: Card::new(tr("0.38", "1.15", "0.38", "0.38", None)),
    },
    OrEndpoint {
        host: "ModelRun",
        tag: "modelrun/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.75", "1", "0.2", "0.75", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.75", "1", "0.25", "0.75", None)),
    },
];

/// `openai/gpt-4.1` (`GET /api/v1/models/openai/gpt-4.1/endpoints`).
static OR_OPENAI_GPT_4_1: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "8", "0.5", "2", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "8", "0.5", "2", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/swedencentral",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "8.8", "0.55", "2.2", None))
        },
    },
];

/// `openai/gpt-4.1-mini` (`GET /api/v1/models/openai/gpt-4.1-mini/endpoints`).
static OR_OPENAI_GPT_4_1_MINI: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.4", "1.6", "0.1", "0.4", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.4", "1.6", "0.1", "0.4", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/swedencentral",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.44", "1.76", "0.11", "0.44", None))
        },
    },
];

/// `openai/gpt-4o` (`GET /api/v1/models/openai/gpt-4o/endpoints`).
static OR_OPENAI_GPT_4O: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card::new(tr("2.5", "10", "2.5", "2.5", None)),
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card::new(tr("2.5", "10", "1.25", "2.5", None)),
    },
];

/// `openai/gpt-4o-mini` (`GET /api/v1/models/openai/gpt-4o-mini/endpoints`).
static OR_OPENAI_GPT_4O_MINI: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.6", "0.075", "0.15", None)),
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.6", "0.075", "0.15", None)),
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/swedencentral",
        class: Class::Standard,
        card: Card::new(tr("0.165", "0.66", "0.0825", "0.165", None)),
    },
];

/// `openai/gpt-5` (`GET /api/v1/models/openai/gpt-5/endpoints`).
static OR_OPENAI_GPT_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.25", "10", "0.125", "1.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/default",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.25", "10", "0.125", "1.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/swedencentral",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.375", "11", "0.1375", "1.375", None))
        },
    },
];

/// `openai/gpt-5-mini` (`GET /api/v1/models/openai/gpt-5-mini/endpoints`).
static OR_OPENAI_GPT_5_MINI: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.125", "1", "0.0125", "0.125", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.25", "2", "0.03", "0.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.25", "2", "0.025", "0.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/swedencentral",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.275", "2.2", "0.033", "0.275", None))
        },
    },
];

/// `openai/gpt-5-nano` (`GET /api/v1/models/openai/gpt-5-nano/endpoints`).
static OR_OPENAI_GPT_5_NANO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.025", "0.2", "0.0025", "0.025", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.05", "0.4", "0.01", "0.05", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.05", "0.4", "0.005", "0.05", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/swedencentral",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.055", "0.44", "0.011", "0.055", None))
        },
    },
];

/// `openai/gpt-5-pro` (`GET /api/v1/models/openai/gpt-5-pro/endpoints`).
static OR_OPENAI_GPT_5_PRO: &[OrEndpoint] = &[OrEndpoint {
    host: "OpenAI",
    tag: "openai",
    class: Class::Standard,
    card: Card {
        tools: OR_SEARCH_10,
        ..Card::new(tr("15", "120", "15", "15", None))
    },
}];

/// `openai/gpt-5.1` (`GET /api/v1/models/openai/gpt-5.1/endpoints`).
static OR_OPENAI_GPT_5_1: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.625", "5", "0.0625", "0.625", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.25", "10", "0.13", "1.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.25", "10", "0.125", "1.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/swedencentral",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.375", "11", "0.143", "1.375", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.5", "20", "0.25", "2.5", None))
        },
    },
];

/// `openai/gpt-5.2` (`GET /api/v1/models/openai/gpt-5.2/endpoints`).
static OR_OPENAI_GPT_5_2: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.875", "7", "0.0875", "0.875", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.75", "14", "0.175", "1.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.75", "14", "0.175", "1.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3.5", "28", "0.35", "3.5", None))
        },
    },
];

/// `openai/gpt-5.2-pro` (`GET /api/v1/models/openai/gpt-5.2-pro/endpoints`).
static OR_OPENAI_GPT_5_2_PRO: &[OrEndpoint] = &[OrEndpoint {
    host: "OpenAI",
    tag: "openai",
    class: Class::Standard,
    card: Card {
        tools: OR_SEARCH_10,
        ..Card::new(tr("21", "168", "21", "21", None))
    },
}];

/// `openai/gpt-5.3-codex` (`GET /api/v1/models/openai/gpt-5.3-codex/endpoints`).
static OR_OPENAI_GPT_5_3_CODEX: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.75", "14", "0.175", "1.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.75", "14", "0.175", "1.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("3.5", "28", "0.35", "3.5", None))
        },
    },
];

/// `openai/gpt-5.4` (`GET /api/v1/models/openai/gpt-5.4/endpoints`).
static OR_OPENAI_GPT_5_4: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2.5", "11.25", "0.25", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.25", "7.5", "0.125", "1.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("5", "22.5", "0.5", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.5", "15", "0.25", "2.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("5", "22.5", "0.5", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.5", "15", "0.25", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("5.5", "24.75", "0.55", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.75", "16.5", "0.275", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("5.5", "24.75", "0.55", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.75", "16.5", "0.275", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("5.5", "24.75", "0.55", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.75", "16.5", "0.275", "2.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("10", "45", "1", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "30", "0.5", "5", None))
        },
    },
];

/// `openai/gpt-5.4-mini` (`GET /api/v1/models/openai/gpt-5.4-mini/endpoints`).
static OR_OPENAI_GPT_5_4_MINI: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.375", "2.25", "0.0375", "0.375", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.75", "4.5", "0.075", "0.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.75", "4.5", "0.075", "0.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.825", "4.95", "0.0825", "0.825", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("1.5", "9", "0.15", "1.5", None))
        },
    },
];

/// `openai/gpt-5.4-nano` (`GET /api/v1/models/openai/gpt-5.4-nano/endpoints`).
static OR_OPENAI_GPT_5_4_NANO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.1", "0.625", "0.01", "0.1", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.2", "1.25", "0.02", "0.2", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.2", "1.25", "0.02", "0.2", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.22", "1.375", "0.022", "0.22", None))
        },
    },
];

/// `openai/gpt-5.4-pro` (`GET /api/v1/models/openai/gpt-5.4-pro/endpoints`).
static OR_OPENAI_GPT_5_4_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("30", "135", "30", "30", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("15", "90", "15", "15", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("60", "270", "60", "60", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("30", "180", "30", "30", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("60", "270", "60", "60", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("30", "180", "30", "30", None))
        },
    },
];

/// `openai/gpt-5.5` (`GET /api/v1/models/openai/gpt-5.5/endpoints`).
static OR_OPENAI_GPT_5_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("5", "22.5", "0.5", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.5", "15", "0.25", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("10", "45", "1", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "30", "0.5", "5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("10", "45", "1", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "30", "0.5", "5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("11", "49.5", "1.1", "11", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "33", "0.55", "5.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("11", "49.5", "1.1", "11", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "33", "0.55", "5.5", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("11", "49.5", "1.1", "11", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("5.5", "33", "0.55", "5.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("25", "112.5", "2.5", "25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("12.5", "75", "1.25", "12.5", None))
        },
    },
];

/// `openai/gpt-5.5-pro` (`GET /api/v1/models/openai/gpt-5.5-pro/endpoints`).
static OR_OPENAI_GPT_5_5_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("30", "135", "30", "30", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("15", "90", "15", "15", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("60", "270", "60", "60", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("30", "180", "30", "30", None))
        },
    },
];

/// `openai/gpt-5.6-luna` (`GET /api/v1/models/openai/gpt-5.6-luna/endpoints`).
static OR_OPENAI_GPT_5_6_LUNA: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("0.2", "0.9", "0.02", "0.25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.1", "0.6", "0.01", "0.125", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.4", "1.8", "0.04", "0.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.2", "1.2", "0.02", "0.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.4", "1.8", "0.04", "0.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.2", "1.2", "0.02", "0.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.44", "1.98", "0.044", "0.55", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.22", "1.32", "0.022", "0.275", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.44", "1.98", "0.044", "0.55", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.22", "1.32", "0.022", "0.275", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.44", "1.98", "0.044", "0.55", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.22", "1.32", "0.022", "0.275", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("0.8", "3.6", "0.08", "1", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.4", "2.4", "0.04", "0.5", None))
        },
    },
];

/// `openai/gpt-5.6-luna-pro` (`GET /api/v1/models/openai/gpt-5.6-luna-pro/endpoints`).
static OR_OPENAI_GPT_5_6_LUNA_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("0.2", "0.9", "0.02", "0.25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.1", "0.6", "0.01", "0.125", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.4", "1.8", "0.04", "0.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.2", "1.2", "0.02", "0.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.4", "1.8", "0.04", "0.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.2", "1.2", "0.02", "0.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.44", "1.98", "0.044", "0.55", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.22", "1.32", "0.022", "0.275", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("0.8", "3.6", "0.08", "1", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.4", "2.4", "0.04", "0.5", None))
        },
    },
];

/// `openai/gpt-5.6-sol` (`GET /api/v1/models/openai/gpt-5.6-sol/endpoints`).
static OR_OPENAI_GPT_5_6_SOL: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2", "7.5", "0.2", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.1", "1.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "30", "0.8", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.4", "5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("8", "30", "0.8", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.4", "5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("8.8", "33", "0.88", "11", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.44", "5.5", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("8.8", "33", "0.88", "11", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.44", "5.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("8.8", "33", "0.88", "11", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.44", "5.5", None))
        },
    },
];

/// `openai/gpt-5.6-sol-pro` (`GET /api/v1/models/openai/gpt-5.6-sol-pro/endpoints`).
static OR_OPENAI_GPT_5_6_SOL_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2", "7.5", "0.2", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.1", "1.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "30", "0.8", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.4", "5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("8", "30", "0.8", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.4", "5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("8.8", "33", "0.88", "11", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4.4", "22", "0.44", "5.5", None))
        },
    },
];

/// `openai/gpt-5.6-terra` (`GET /api/v1/models/openai/gpt-5.6-terra/endpoints`).
static OR_OPENAI_GPT_5_6_TERRA: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2", "9", "0.2", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "6", "0.1", "1.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "18", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "12", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "18", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "12", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "19.8", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "13.2", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "19.8", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "13.2", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "19.8", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "13.2", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "36", "0.8", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "24", "0.4", "5", None))
        },
    },
];

/// `openai/gpt-5.6-terra-pro` (`GET /api/v1/models/openai/gpt-5.6-terra-pro/endpoints`).
static OR_OPENAI_GPT_5_6_TERRA_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2", "9", "0.2", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "6", "0.1", "1.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "18", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "12", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "18", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "12", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "19.8", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "13.2", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "36", "0.8", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "24", "0.4", "5", None))
        },
    },
];

/// `openai/gpt-6-astra` (`GET /api/v1/models/openai/gpt-6-astra/endpoints`).
static OR_OPENAI_GPT_6_ASTRA: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("10", "37.5", "1", "12.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("20", "75", "2", "25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("20", "75", "2", "25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-west-2",
        class: Class::Standard,
        card: Card {
            long: Some(tr("22", "82.5", "2.2", "27.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("11", "55", "1.1", "13.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("22", "82.5", "2.2", "27.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("11", "55", "1.1", "13.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("40", "150", "4", "50", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("20", "100", "2", "25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/ultrafast",
        class: Class::Ultrafast,
        card: Card {
            long: Some(tr("120", "450", "12", "150", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("60", "300", "6", "75", None))
        },
    },
];

/// `openai/gpt-6-astra-pro` (`GET /api/v1/models/openai/gpt-6-astra-pro/endpoints`).
static OR_OPENAI_GPT_6_ASTRA_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("10", "37.5", "1", "12.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("5", "25", "0.5", "6.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("20", "75", "2", "25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("20", "75", "2", "25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("10", "50", "1", "12.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("22", "82.5", "2.2", "27.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("11", "55", "1.1", "13.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("40", "150", "4", "50", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("20", "100", "2", "25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/ultrafast",
        class: Class::Ultrafast,
        card: Card {
            long: Some(tr("120", "450", "12", "150", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("60", "300", "6", "75", None))
        },
    },
];

/// `openai/gpt-6-luna` (`GET /api/v1/models/openai/gpt-6-luna/endpoints`).
static OR_OPENAI_GPT_6_LUNA: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("0.1", "0.375", "0.01", "0.125", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.05", "0.25", "0.005", "0.0625", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.2", "0.75", "0.02", "0.25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.1", "0.5", "0.01", "0.125", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.2", "0.75", "0.02", "0.25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.1", "0.5", "0.01", "0.125", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.22", "0.825", "0.022", "0.275", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.11", "0.55", "0.011", "0.1375", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.22", "0.825", "0.022", "0.275", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.11", "0.55", "0.011", "0.1375", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.22", "0.825", "0.022", "0.275", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.11", "0.55", "0.011", "0.1375", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("0.4", "1.5", "0.04", "0.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.2", "1", "0.02", "0.25", None))
        },
    },
];

/// `openai/gpt-6-luna-pro` (`GET /api/v1/models/openai/gpt-6-luna-pro/endpoints`).
static OR_OPENAI_GPT_6_LUNA_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("0.1", "0.375", "0.01", "0.125", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.05", "0.25", "0.005", "0.0625", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.2", "0.75", "0.02", "0.25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.1", "0.5", "0.01", "0.125", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.2", "0.75", "0.02", "0.25", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.1", "0.5", "0.01", "0.125", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.22", "0.825", "0.022", "0.275", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.11", "0.55", "0.011", "0.1375", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.22", "0.825", "0.022", "0.275", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.11", "0.55", "0.011", "0.1375", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("0.4", "1.5", "0.04", "0.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("0.2", "1", "0.02", "0.25", None))
        },
    },
];

/// `openai/gpt-6-sol` (`GET /api/v1/models/openai/gpt-6-sol/endpoints`).
static OR_OPENAI_GPT_6_SOL: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2", "7.5", "0.2", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.1", "1.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "30", "0.8", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.4", "5", None))
        },
    },
];

/// `openai/gpt-6-sol-pro` (`GET /api/v1/models/openai/gpt-6-sol-pro/endpoints`).
static OR_OPENAI_GPT_6_SOL_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2", "7.5", "0.2", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.1", "1.25", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.4", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.2", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.44", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.22", "2.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "30", "0.8", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.4", "5", None))
        },
    },
];

/// `openai/gpt-6.1-sol` (`GET /api/v1/models/openai/gpt-6.1-sol/endpoints`).
static OR_OPENAI_GPT_6_1_SOL: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2", "7.5", "0.1", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.05", "1.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.2", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.2", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.22", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.11", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.22", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.11", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.22", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.11", "2.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "30", "0.4", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.2", "5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/ultrafast",
        class: Class::Ultrafast,
        card: Card {
            long: Some(tr("24", "90", "1.2", "30", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("12", "60", "0.6", "15", None))
        },
    },
];

/// `openai/gpt-6.1-sol-pro` (`GET /api/v1/models/openai/gpt-6.1-sol-pro/endpoints`).
static OR_OPENAI_GPT_6_1_SOL_PRO: &[OrEndpoint] = &[
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/flex",
        class: Class::Flex,
        card: Card {
            long: Some(tr("2", "7.5", "0.1", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("1", "5", "0.05", "1.25", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.2", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "15", "0.2", "5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2", "10", "0.1", "2.5", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/eu",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.22", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.11", "2.75", None))
        },
    },
    OrEndpoint {
        host: "Azure",
        tag: "azure/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "16.5", "0.22", "5.5", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "11", "0.11", "2.75", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/fast",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "30", "0.4", "10", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("4", "20", "0.2", "5", None))
        },
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai/ultrafast",
        class: Class::Ultrafast,
        card: Card {
            long: Some(tr("24", "90", "1.2", "30", None)),
            long_context: Some(LongContext {
                threshold: 272000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("12", "60", "0.6", "15", None))
        },
    },
];

/// `x-ai/grok-4.20` (`GET /api/v1/models/x-ai/grok-4.20/endpoints`).
static OR_X_AI_GROK_4_20: &[OrEndpoint] = &[
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr",
        class: Class::Standard,
        card: Card {
            long: Some(tr("2.5", "5", "0.4", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("2.5", "5", "0.4", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("5", "10", "0.8", "5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2.5", "5", "0.4", "2.5", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("5", "10", "0.8", "5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2.5", "5", "0.4", "2.5", None))
        },
    },
];

/// `x-ai/grok-4.20-multi-agent` (`GET /api/v1/models/x-ai/grok-4.20-multi-agent/endpoints`).
static OR_X_AI_GROK_4_20_MULTI_AGENT: &[OrEndpoint] = &[
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr",
        class: Class::Standard,
        card: Card {
            long: Some(tr("2.5", "5", "0.4", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("2.5", "5", "0.4", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("5", "10", "0.8", "5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2.5", "5", "0.4", "2.5", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("5", "10", "0.8", "5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2.5", "5", "0.4", "2.5", None))
        },
    },
];

/// `x-ai/grok-4.3` (`GET /api/v1/models/x-ai/grok-4.3/endpoints`).
static OR_X_AI_GROK_4_3: &[OrEndpoint] = &[
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr",
        class: Class::Standard,
        card: Card {
            long: Some(tr("2.5", "5", "0.4", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("2.5", "5", "0.4", "2.5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("1.25", "2.5", "0.2", "1.25", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("5", "10", "0.8", "5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2.5", "5", "0.4", "2.5", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("5", "10", "0.8", "5", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2.5", "5", "0.4", "2.5", None))
        },
    },
];

/// `x-ai/grok-4.5` (`GET /api/v1/models/x-ai/grok-4.5/endpoints`).
static OR_X_AI_GROK_4_5: &[OrEndpoint] = &[
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "12", "0.6", "4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2", "6", "0.3", "2", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "12", "0.6", "4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2", "6", "0.3", "2", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "24", "1.2", "8", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("4", "12", "0.6", "4", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "24", "1.2", "8", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("4", "12", "0.6", "4", None))
        },
    },
];

/// `x-ai/grok-4.6` (`GET /api/v1/models/x-ai/grok-4.6/endpoints`).
static OR_X_AI_GROK_4_6: &[OrEndpoint] = &[
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "12", "1", "4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2", "6", "0.5", "2", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "12", "1", "4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2", "6", "0.5", "2", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "13.2", "1.1", "4.4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2.2", "6.6", "0.55", "2.2", None))
        },
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-west-2",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "13.2", "1.1", "0", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_10,
            ..Card::new(tr("2.2", "6.6", "0.55", "0", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "24", "2", "8", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("4", "12", "1", "4", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "24", "2", "8", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("4", "12", "1", "4", None))
        },
    },
];

/// `x-ai/grok-4.7` (`GET /api/v1/models/x-ai/grok-4.7/endpoints`).
static OR_X_AI_GROK_4_7: &[OrEndpoint] = &[
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "12", "1", "4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2", "6", "0.5", "2", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4", "12", "1", "4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2", "6", "0.5", "2", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/us",
        class: Class::Standard,
        card: Card {
            long: Some(tr("4.4", "13.2", "1.1", "4.4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2.2", "6.6", "0.55", "2.2", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "24", "2", "8", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("4", "12", "1", "4", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("8", "24", "2", "8", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("4", "12", "1", "4", None))
        },
    },
];

/// `x-ai/grok-build-0.1` (`GET /api/v1/models/x-ai/grok-build-0.1/endpoints`).
static OR_X_AI_GROK_BUILD_0_1: &[OrEndpoint] = &[
    OrEndpoint {
        host: "xAI",
        tag: "xai",
        class: Class::Standard,
        card: Card {
            long: Some(tr("2", "4", "0.4", "2", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("1", "2", "0.2", "1", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr",
        class: Class::Standard,
        card: Card {
            long: Some(tr("2", "4", "0.4", "2", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("1", "2", "0.2", "1", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/zdr/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("4", "8", "0.8", "4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2", "4", "0.4", "2", None))
        },
    },
    OrEndpoint {
        host: "xAI",
        tag: "xai/priority",
        class: Class::Fast,
        card: Card {
            long: Some(tr("4", "8", "0.8", "4", None)),
            long_context: Some(LongContext {
                threshold: 200000,
                inclusive: false,
            }),
            tools: OR_SEARCH_5,
            ..Card::new(tr("2", "4", "0.4", "2", None))
        },
    },
];

/// `meta-llama/llama-3.1-8b-instruct` (`GET /api/v1/models/meta-llama/llama-3.1-8b-instruct/endpoints`).
static OR_META_LLAMA_LLAMA_3_1_8B_INSTRUCT: &[OrEndpoint] = &[
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.02", "0.04", "0.02", "0.02", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.02", "0.05", "0.02", "0.02", None)),
    },
    OrEndpoint {
        host: "Groq",
        tag: "groq",
        class: Class::Standard,
        card: Card::new(tr("0.05", "0.08", "0.025", "0.05", None)),
    },
    OrEndpoint {
        host: "Cloudflare",
        tag: "cloudflare/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.152", "0.287", "0.152", "0.152", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.22", "0.22", "0.22", "0.22", None)),
    },
];

/// `meta-llama/llama-3.3-70b-instruct` (`GET /api/v1/models/meta-llama/llama-3.3-70b-instruct/endpoints`).
static OR_META_LLAMA_LLAMA_3_3_70B_INSTRUCT: &[OrEndpoint] = &[
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/turbo",
        class: Class::Standard,
        card: Card::new(tr("0.1", "0.32", "0.1", "0.1", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.135", "0.4", "0.135", "0.135", None)),
    },
    OrEndpoint {
        host: "AkashML",
        tag: "akashml/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.2", "0.52", "0.1", "0.2", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.22", "0.5", "0.11", "0.22", None)),
    },
    OrEndpoint {
        host: "Cloudflare",
        tag: "cloudflare/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.293", "2.253", "0.293", "0.293", None)),
    },
    OrEndpoint {
        host: "SambaNova",
        tag: "sambanova-turbo",
        class: Class::Standard,
        card: Card::new(tr("0.45", "0.9", "0.45", "0.45", None)),
    },
    OrEndpoint {
        host: "Groq",
        tag: "groq",
        class: Class::Standard,
        card: Card::new(tr("0.59", "0.79", "0.295", "0.59", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp16",
        class: Class::Standard,
        card: Card::new(tr("0.71", "0.71", "0.71", "0.71", None)),
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us-central1",
        class: Class::Standard,
        card: Card::new(tr("0.72", "0.72", "0.72", "0.72", None)),
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex",
        class: Class::Standard,
        card: Card::new(tr("0.72", "0.72", "0.72", "0.72", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("1.04", "1.04", "1.04", "1.04", None)),
    },
];

/// `meta-llama/llama-4-maverick` (`GET /api/v1/models/meta-llama/llama-4-maverick/endpoints`).
static OR_META_LLAMA_LLAMA_4_MAVERICK: &[OrEndpoint] = &[
    OrEndpoint {
        host: "DigitalOcean",
        tag: "digitalocean",
        class: Class::Standard,
        card: Card::new(tr("0.1875", "0.6525", "0.05", "0.1875", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.27", "0.85", "0.27", "0.27", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.35", "1", "0.17", "0.35", None)),
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us-east5",
        class: Class::Standard,
        card: Card::new(tr("0.35", "1.15", "0.35", "0.35", None)),
    },
];

/// `meta-llama/llama-4-scout` (`GET /api/v1/models/meta-llama/llama-4-scout/endpoints`).
static OR_META_LLAMA_LLAMA_4_SCOUT: &[OrEndpoint] = &[
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.1", "0.3", "0.1", "0.1", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.18", "0.59", "0.18", "0.18", None)),
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us-east5",
        class: Class::Standard,
        card: Card::new(tr("0.25", "0.7", "0.25", "0.25", None)),
    },
];

/// `meta/muse-glimmer-30b` (`GET /api/v1/models/meta/muse-glimmer-30b/endpoints`).
static OR_META_MUSE_GLIMMER_30B: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Phala",
        tag: "phala",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.1", "0.04", "0.3", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.04", "0.3", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("0.35", "1.5", "0.04", "0.35", None)),
    },
];

/// `minimax/minimax-m3` (`GET /api/v1/models/minimax/minimax-m3/endpoints`).
static OR_MINIMAX_MINIMAX_M3: &[OrEndpoint] = &[
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.23", "0.96", "0.05", "0.23", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.24", "0.96", "0.048", "0.24", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.28", "1.1", "0.056", "0.28", None)),
    },
    OrEndpoint {
        host: "StreamLake",
        tag: "streamlake/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.06", "0.3", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.06", "0.3", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.06", "0.3", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.06", "0.3", None)),
    },
    OrEndpoint {
        host: "AtlasCloud",
        tag: "atlas-cloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.06", "0.3", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.06", "0.3", None)),
    },
    OrEndpoint {
        host: "Minimax",
        tag: "minimax/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.3", "1.2", "0.06", "0.3", None)),
    },
    OrEndpoint {
        host: "Mara",
        tag: "mara",
        class: Class::Standard,
        card: Card::new(tr("0.6", "2.4", "0.6", "0.6", None)),
    },
    OrEndpoint {
        host: "SambaNova",
        tag: "sambanova",
        class: Class::Standard,
        card: Card::new(tr("0.6", "2.4", "0.06", "0.6", None)),
    },
    OrEndpoint {
        host: "ModelRun",
        tag: "modelrun/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.75", "3", "0.15", "0.75", None)),
    },
];

/// `moonshotai/kimi-k2.6` (`GET /api/v1/models/moonshotai/kimi-k2.6/endpoints`).
static OR_MOONSHOTAI_KIMI_K2_6: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Baidu",
        tag: "baidu/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.437", "1.84", "0.0736", "0.437", None)),
    },
    OrEndpoint {
        host: "Inceptron",
        tag: "inceptron/int4",
        class: Class::Standard,
        card: Card::new(tr("0.4375", "2.45", "0.1087", "0.4375", None)),
    },
    OrEndpoint {
        host: "Chutes",
        tag: "chutes/int4",
        class: Class::Standard,
        card: Card::new(tr("0.5", "2.85", "0.05", "0.5", None)),
    },
    OrEndpoint {
        host: "DigitalOcean",
        tag: "digitalocean",
        class: Class::Standard,
        card: Card::new(tr("0.57", "2.4", "0.114", "0.57", None)),
    },
    OrEndpoint {
        host: "StreamLake",
        tag: "streamlake/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.5985", "2.52", "0.1008", "0.5985", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.65", "3.41", "0.15", "0.65", None)),
    },
    OrEndpoint {
        host: "Crusoe",
        tag: "crusoe/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.7", "3.5", "0.35", "0.7", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.75", "3.5", "0.15", "0.75", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/int4",
        class: Class::Standard,
        card: Card::new(tr("0.75", "3.5", "0.16", "0.75", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/int4",
        class: Class::Standard,
        card: Card::new(tr("0.75", "3.5", "0.16", "0.75", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.77", "3.4", "0.14", "0.77", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita",
        class: Class::Standard,
        card: Card::new(tr("0.8", "3.4", "0.16", "0.8", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.855", "3.6", "0.144", "0.855", None)),
    },
    OrEndpoint {
        host: "AtlasCloud",
        tag: "atlas-cloud/int4",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4", "0.16", "0.95", None)),
    },
    OrEndpoint {
        host: "Cloudflare",
        tag: "cloudflare",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4", "0.16", "0.95", None)),
    },
    OrEndpoint {
        host: "Moonshot AI",
        tag: "moonshotai/int4",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4", "0.16", "0.95", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala",
        class: Class::Standard,
        card: Card::new(tr("1.09", "4.6", "0.37", "1.09", None)),
    },
];

/// `moonshotai/kimi-k2.7-code` (`GET /api/v1/models/moonshotai/kimi-k2.7-code/endpoints`).
static OR_MOONSHOTAI_KIMI_K2_7_CODE: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Inceptron",
        tag: "inceptron/int4",
        class: Class::Standard,
        card: Card::new(tr("0.6712", "3.35", "0.18", "0.6712", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/int4",
        class: Class::Standard,
        card: Card::new(tr("0.71", "3.5", "0.15", "0.71", None)),
    },
    OrEndpoint {
        host: "StreamLake",
        tag: "streamlake",
        class: Class::Standard,
        card: Card::new(tr("0.7125", "3", "0.1425", "0.7125", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/int4",
        class: Class::Standard,
        card: Card::new(tr("0.75", "3.5", "0.16", "0.75", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.85916", "3.8", "0.17993", "0.85916", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/int4",
        class: Class::Standard,
        card: Card::new(tr("0.912", "3.84", "0.1824", "0.912", None)),
    },
    OrEndpoint {
        host: "Nebius",
        tag: "nebius/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4", "0.95", "0.95", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4", "0.19", "0.95", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4", "0.19", "0.95", None)),
    },
    OrEndpoint {
        host: "Cloudflare",
        tag: "cloudflare",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4", "0.19", "0.95", None)),
    },
    OrEndpoint {
        host: "Moonshot AI",
        tag: "moonshotai/int4",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4", "0.19", "0.95", None)),
    },
    OrEndpoint {
        host: "Moonshot AI",
        tag: "moonshotai/highspeed",
        class: Class::Standard,
        card: Card::new(tr("1.9", "8", "0.38", "1.9", None)),
    },
];

/// `moonshotai/kimi-k3` (`GET /api/v1/models/moonshotai/kimi-k3/endpoints`).
static OR_MOONSHOTAI_KIMI_K3: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Wafer",
        tag: "wafer",
        class: Class::Standard,
        card: Card::new(tr("0.31", "14.89", "0.3", "0.31", None)),
    },
    OrEndpoint {
        host: "Morph",
        tag: "morph/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.312", "14.9", "0.29", "0.312", None)),
    },
    OrEndpoint {
        host: "InferenceNet",
        tag: "inference-net/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.5", "13.5", "0.28", "0.5", None)),
    },
    OrEndpoint {
        host: "Sail Research",
        tag: "sail-research/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.84", "13.5", "0.3", "0.84", None)),
    },
    OrEndpoint {
        host: "AkashML",
        tag: "akashml/fp4",
        class: Class::Standard,
        card: Card::new(tr("1.2", "14", "1.2", "1.2", None)),
    },
    OrEndpoint {
        host: "Makora",
        tag: "makora/fp4",
        class: Class::Standard,
        card: Card::new(tr("1.53", "12.75", "0.204", "1.53", None)),
    },
    OrEndpoint {
        host: "Decart",
        tag: "decart/mxfp4",
        class: Class::Standard,
        card: Card::new(tr("1.92", "9.6", "0.192", "1.92", None)),
    },
    OrEndpoint {
        host: "Relace",
        tag: "relace/fp4",
        class: Class::Standard,
        card: Card::new(tr("2", "14", "0.3", "2", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala",
        class: Class::Standard,
        card: Card::new(tr("2.55", "12.75", "0.255", "2.55", None)),
    },
    OrEndpoint {
        host: "DigitalOcean",
        tag: "digitalocean",
        class: Class::Standard,
        card: Card::new(tr("2.55", "12.95", "0.255", "2.55", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp4",
        class: Class::Standard,
        card: Card::new(tr("2.6", "13", "0.26", "2.6", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("2.7", "13.5", "0.27", "2.7", None)),
    },
    OrEndpoint {
        host: "Wafer",
        tag: "wafer/us",
        class: Class::Standard,
        card: Card::new(tr("2.8", "14", "0.3", "2.8", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/mxfp4",
        class: Class::Standard,
        card: Card::new(tr("2.85", "14.25", "0.285", "2.85", None)),
    },
    OrEndpoint {
        host: "InferenceNet",
        tag: "inference-net/fast",
        class: Class::Fast,
        card: Card::new(tr("2.99", "15", "0.45", "2.99", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fast",
        class: Class::Fast,
        card: Card::new(tr("3", "15", "0.3", "3", None)),
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/us-east-2",
        class: Class::Standard,
        card: Card::new(tr("3", "15", "0.3", "3.75", None)),
    },
    OrEndpoint {
        host: "Chutes",
        tag: "chutes/mxfp4",
        class: Class::Standard,
        card: Card::new(tr("3", "15", "0.3", "3", None)),
    },
    OrEndpoint {
        host: "Modal",
        tag: "modal/mxfp4",
        class: Class::Standard,
        card: Card::new(tr("3", "15", "0.3", "3", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks",
        class: Class::Standard,
        card: Card::new(tr("3", "15", "0.3", "3", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp8",
        class: Class::Standard,
        card: Card::new(tr("3", "15", "0.3", "3", None)),
    },
    OrEndpoint {
        host: "Moonshot AI",
        tag: "moonshotai/mxfp4",
        class: Class::Standard,
        card: Card::new(tr("3", "15", "0.3", "3", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card::new(tr("3.45", "17.25", "0.345", "3.45", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks/us",
        class: Class::Standard,
        card: Card::new(tr("4.5", "22.5", "0.45", "4.5", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks/fast",
        class: Class::Fast,
        card: Card::new(tr("4.5", "22.5", "0.45", "4.5", None)),
    },
];

/// `openai/o3` (`GET /api/v1/models/openai/o3/endpoints`).
static OR_OPENAI_O3: &[OrEndpoint] = &[OrEndpoint {
    host: "OpenAI",
    tag: "openai",
    class: Class::Standard,
    card: Card {
        tools: OR_SEARCH_10,
        ..Card::new(tr("2", "8", "0.5", "2", None))
    },
}];

/// `openai/o3-pro` (`GET /api/v1/models/openai/o3-pro/endpoints`).
static OR_OPENAI_O3_PRO: &[OrEndpoint] = &[OrEndpoint {
    host: "OpenAI",
    tag: "openai",
    class: Class::Standard,
    card: Card {
        tools: OR_SEARCH_10,
        ..Card::new(tr("20", "80", "20", "20", None))
    },
}];

/// `openai/gpt-oss-20b` (`GET /api/v1/models/openai/gpt-oss-20b/endpoints`).
static OR_OPENAI_GPT_OSS_20B: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Darkbloom",
        tag: "darkbloom/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.018", "0.09", "0.009", "0.018", None)),
    },
    OrEndpoint {
        host: "AkashML",
        tag: "akashml/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.02", "0.1", "0.02", "0.02", None)),
    },
    OrEndpoint {
        host: "DekaLLM",
        tag: "dekallm/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.029", "0.14", "0.029", "0.029", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.03", "0.13", "0.03", "0.03", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.03", "0.14", "0.03", "0.03", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.03", "0.15", "0.02", "0.03", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.04", "0.18", "0.04", "0.04", None)),
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock/eu-west-1",
        class: Class::Standard,
        card: Card::new(tr("0.07", "0.15", "0.07", "0.07", None)),
    },
    OrEndpoint {
        host: "Amazon Bedrock",
        tag: "amazon-bedrock",
        class: Class::Standard,
        card: Card::new(tr("0.07", "0.15", "0.07", "0.07", None)),
    },
    OrEndpoint {
        host: "Google",
        tag: "google-vertex/us-central1",
        class: Class::Standard,
        card: Card::new(tr("0.07", "0.25", "0.07", "0.07", None)),
    },
    OrEndpoint {
        host: "Groq",
        tag: "groq",
        class: Class::Standard,
        card: Card::new(tr("0.075", "0.3", "0.0375", "0.075", None)),
    },
];

/// `openai/gpt-oss-safeguard-20b` (`GET /api/v1/models/openai/gpt-oss-safeguard-20b/endpoints`).
static OR_OPENAI_GPT_OSS_SAFEGUARD_20B: &[OrEndpoint] = &[OrEndpoint {
    host: "Groq",
    tag: "groq",
    class: Class::Standard,
    card: Card::new(tr("0.075", "0.3", "0.0375", "0.075", None)),
}];

/// `qwen/qwen-2.5-7b-instruct` (`GET /api/v1/models/qwen/qwen-2.5-7b-instruct/endpoints`).
static OR_QWEN_QWEN_2_5_7B_INSTRUCT: &[OrEndpoint] = &[OrEndpoint {
    host: "Phala",
    tag: "phala",
    class: Class::Standard,
    card: Card::new(tr("0.1", "0.2", "0.1", "0.1", None)),
}];

/// `qwen/qwen3.5-9b` (`GET /api/v1/models/qwen/qwen3.5-9b/endpoints`).
static OR_QWEN_QWEN3_5_9B: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Darkbloom",
        tag: "darkbloom/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.08", "0.13", "0.04", "0.08", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.1", "0.15", "0.1", "0.1", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.1", "0.15", "0.1", "0.1", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.1", "0.15", "0.1", "0.1", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.1", "0.25", "0.1", "0.1", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("0.17", "0.25", "0.17", "0.17", None)),
    },
];

/// `qwen/qwen3.6-plus` (`GET /api/v1/models/qwen/qwen3.6-plus/endpoints`).
static OR_QWEN_QWEN3_6_PLUS: &[OrEndpoint] = &[OrEndpoint {
    host: "Alibaba",
    tag: "alibaba",
    class: Class::Standard,
    card: Card {
        long: Some(tr("1.3", "3.9", "1.3", "1.625", None)),
        long_context: Some(LongContext {
            threshold: 256000,
            inclusive: false,
        }),
        ..Card::new(tr("0.325", "1.95", "0.325", "0.40625", None))
    },
}];

/// `qwen/qwen3.7-max` (`GET /api/v1/models/qwen/qwen3.7-max/endpoints`).
static OR_QWEN_QWEN3_7_MAX: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card::new(tr("1.475", "4.425", "0.295", "1.84375", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/us-east-1",
        class: Class::Standard,
        card: Card::new(tr("2", "6", "0.4", "2", None)),
    },
];

/// `qwen/qwen3.7-plus` (`GET /api/v1/models/qwen/qwen3.7-plus/endpoints`).
static OR_QWEN_QWEN3_7_PLUS: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/us-east-1",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.96", "3.84", "0.192", "1.2", None)),
            long_context: Some(LongContext {
                threshold: 256000,
                inclusive: false,
            }),
            ..Card::new(tr("0.32", "1.28", "0.064", "0.4", None))
        },
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card {
            long: Some(tr("0.96", "3.84", "0.192", "1.2", None)),
            long_context: Some(LongContext {
                threshold: 256000,
                inclusive: false,
            }),
            ..Card::new(tr("0.32", "1.28", "0.064", "0.4", None))
        },
    },
];

/// `qwen/qwen3.8-2.4t-a95b` (`GET /api/v1/models/qwen/qwen3.8-2.4t-a95b/endpoints`).
static OR_QWEN_QWEN3_8_2_4T_A95B: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Novita",
        tag: "novita",
        class: Class::Standard,
        card: Card::new(tr("2", "6", "0.25", "2", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card::new(tr("2", "6", "0.25", "2.5", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("2", "6", "0.25", "2", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice",
        class: Class::Standard,
        card: Card::new(tr("2", "6", "0.25", "2", None)),
    },
    OrEndpoint {
        host: "Modal",
        tag: "modal",
        class: Class::Standard,
        card: Card::new(tr("2", "6", "0.25", "2", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp4",
        class: Class::Standard,
        card: Card::new(tr("2", "6", "0.2", "2", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("2", "6", "0.25", "2", None)),
    },
];

/// `qwen/qwen3.8-27b` (`GET /api/v1/models/qwen/qwen3.8-27b/endpoints`).
static OR_QWEN_QWEN3_8_27B: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Near AI",
        tag: "near-ai/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.04", "1.35", "0.018", "0.04", None)),
    },
    OrEndpoint {
        host: "Wafer",
        tag: "wafer",
        class: Class::Standard,
        card: Card::new(tr("0.04", "2.3", "0.0198", "0.04", None)),
    },
    OrEndpoint {
        host: "DekaLLM",
        tag: "dekallm",
        class: Class::Standard,
        card: Card::new(tr("0.049", "3", "0.02", "0.049", None)),
    },
    OrEndpoint {
        host: "Darkbloom",
        tag: "darkbloom/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.05", "2.2", "0.025", "0.05", None)),
    },
    OrEndpoint {
        host: "Ionstream",
        tag: "ionstream/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.0893", "2.35", "0.085", "0.0893", None)),
    },
    OrEndpoint {
        host: "Reka",
        tag: "reka/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.14", "1.4", "0.035", "0.14", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala",
        class: Class::Standard,
        card: Card::new(tr("0.15", "1.875", "0.0375", "0.15", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/bf16",
        class: Class::Standard,
        card: Card::new(tr("0.15", "1.875", "0.0375", "0.15", None)),
    },
    OrEndpoint {
        host: "Mancer 2",
        tag: "mancer/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.2", "2.5", "0.2", "0.2", None)),
    },
    OrEndpoint {
        host: "AkashML",
        tag: "akashml/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.225", "1.98", "0.05", "0.225", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.24", "2.2", "0.05", "0.24", None)),
    },
    OrEndpoint {
        host: "Chutes",
        tag: "chutes/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.24", "2.2", "0.024", "0.24", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.4", "3", "0.15", "0.4", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita",
        class: Class::Standard,
        card: Card::new(tr("0.42", "3", "0.085", "0.42", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card::new(tr("0.425", "2.55", "0.085", "0.53125", None)),
    },
    OrEndpoint {
        host: "Cloudflare",
        tag: "cloudflare",
        class: Class::Standard,
        card: Card::new(tr("0.45", "3.2", "0.05", "0.45", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.45", "3.2", "0.45", "0.45", None)),
    },
    OrEndpoint {
        host: "ModelRun",
        tag: "modelrun/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.7", "4.7", "0.135", "0.7", None)),
    },
    OrEndpoint {
        host: "Cerebras",
        tag: "cerebras/fp16",
        class: Class::Standard,
        card: Card::new(tr("0.99", "1.49", "0.99", "0.99", None)),
    },
];

/// `qwen/qwen3.8-flash` (`GET /api/v1/models/qwen/qwen3.8-flash/endpoints`).
static OR_QWEN_QWEN3_8_FLASH: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/us-east-1",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.47", "0.016", "0.15", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.47", "0.016", "0.2", None)),
    },
];

/// `openai/text-embedding-3-large` (`GET /api/v1/models/openai/text-embedding-3-large/endpoints`).
static OR_OPENAI_TEXT_EMBEDDING_3_LARGE: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card::new(tr("0.13", "0", "0.13", "0.13", None)),
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card::new(tr("0.13", "0", "0.13", "0.13", None)),
    },
];

/// `openai/text-embedding-3-small` (`GET /api/v1/models/openai/text-embedding-3-small/endpoints`).
static OR_OPENAI_TEXT_EMBEDDING_3_SMALL: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Azure",
        tag: "azure",
        class: Class::Standard,
        card: Card::new(tr("0.02", "0", "0.02", "0.02", None)),
    },
    OrEndpoint {
        host: "OpenAI",
        tag: "openai",
        class: Class::Standard,
        card: Card::new(tr("0.02", "0", "0.02", "0.02", None)),
    },
];

/// `thinkingmachines/inkling` (`GET /api/v1/models/thinkingmachines/inkling/endpoints`).
static OR_THINKINGMACHINES_INKLING: &[OrEndpoint] = &[
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.95", "4.05", "0.16", "0.95", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("1", "4.05", "0.17", "1", None)),
    },
];

/// `z-ai/glm-5.1` (`GET /api/v1/models/z-ai/glm-5.1/endpoints`).
static OR_Z_AI_GLM_5_1: &[OrEndpoint] = &[
    OrEndpoint {
        host: "StreamLake",
        tag: "streamlake/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.966", "3.036", "0.1794", "0.966", None)),
    },
    OrEndpoint {
        host: "Chutes",
        tag: "chutes/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.98", "3.08", "0.098", "0.98", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.19", "3.74", "0.6", "1.19", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala",
        class: Class::Standard,
        card: Card::new(tr("1.21", "4.2", "0.6", "1.21", None)),
    },
    OrEndpoint {
        host: "AtlasCloud",
        tag: "atlas-cloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.26", "3.96", "0.234", "1.26", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.33", "4.18", "0.247", "1.33", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.38", "4.4", "0.26", "1.38", None)),
    },
    OrEndpoint {
        host: "Nebius",
        tag: "nebius/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "1.4", "1.4", None)),
    },
    OrEndpoint {
        host: "Baidu",
        tag: "baidu/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Friendli",
        tag: "friendli",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Z.AI",
        tag: "z-ai/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4014", "4.4044", "0.26026", "1.4014", None)),
    },
];

/// `z-ai/glm-5.2` (`GET /api/v1/models/z-ai/glm-5.2/endpoints`).
static OR_Z_AI_GLM_5_2: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Wafer",
        tag: "wafer/us",
        class: Class::Standard,
        card: Card::new(tr("0.06", "7", "0.059", "0.06", None)),
    },
    OrEndpoint {
        host: "InferenceNet",
        tag: "inference-net/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.18", "4.4", "0.12", "0.18", None)),
    },
    OrEndpoint {
        host: "Morph",
        tag: "morph/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.18", "6", "0.137", "0.18", None)),
    },
    OrEndpoint {
        host: "Wafer",
        tag: "wafer",
        class: Class::Standard,
        card: Card::new(tr("0.19", "10", "0.18", "0.19", None)),
    },
    OrEndpoint {
        host: "Decart",
        tag: "decart/mxfp4",
        class: Class::Standard,
        card: Card::new(tr("0.273", "1.68", "0.105", "0.273", None)),
    },
    OrEndpoint {
        host: "Cloudflare",
        tag: "cloudflare",
        class: Class::Standard,
        card: Card::new(tr("0.5", "6", "0.26", "0.5", None)),
    },
    OrEndpoint {
        host: "StreamLake",
        tag: "streamlake/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.55552", "1.74592", "0.103168", "0.55552", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.5625", "1.8", "0.105", "0.5625", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.6496", "2.0416", "0.12064", "0.6496", None)),
    },
    OrEndpoint {
        host: "DigitalOcean",
        tag: "digitalocean",
        class: Class::Standard,
        card: Card::new(tr("0.7", "2.2", "0.105", "0.7", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.76", "2.42", "0.14", "0.76", None)),
    },
    OrEndpoint {
        host: "AtlasCloud",
        tag: "atlas-cloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.938", "2.948", "0.1742", "0.938", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.966", "3.036", "0.1932", "0.966", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.12", "3.52", "0.224", "1.12", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.19", "3.74", "0.221", "1.19", None)),
    },
    OrEndpoint {
        host: "Inceptron",
        tag: "inceptron/fp4",
        class: Class::Standard,
        card: Card::new(tr("1.25", "5.46", "0.2", "1.25", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.26", "3", "0.22", "1.26", None)),
    },
    OrEndpoint {
        host: "Nebius",
        tag: "nebius/fp4",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "1.4", "1.4", None)),
    },
    OrEndpoint {
        host: "Mistral",
        tag: "mistral/zdr",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.14", "1.4", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.14", "1.4", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Baidu",
        tag: "baidu/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.14", "1.4", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp4",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Friendli",
        tag: "friendli",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Z.AI",
        tag: "z-ai/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Mistral",
        tag: "mistral/eu",
        class: Class::Standard,
        card: Card::new(tr("1.54", "4.84", "0.154", "1.54", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fast",
        class: Class::Fast,
        card: Card::new(tr("2.1", "6.6", "0.21", "2.1", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fast",
        class: Class::Fast,
        card: Card::new(tr("2.1", "6.6", "0.21", "2.1", None)),
    },
    OrEndpoint {
        host: "Baidu",
        tag: "baidu/fast",
        class: Class::Fast,
        card: Card::new(tr("2.25", "7.88", "0.56", "2.25", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/fast",
        class: Class::Fast,
        card: Card::new(tr("2.31", "7.26", "0.462", "2.31", None)),
    },
    OrEndpoint {
        host: "Decart",
        tag: "decart/fast",
        class: Class::Fast,
        card: Card::new(tr("2.5", "9", "0.54", "2.5", None)),
    },
];

/// `z-ai/glm-5.3` (`GET /api/v1/models/z-ai/glm-5.3/endpoints`).
static OR_Z_AI_GLM_5_3: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Wafer",
        tag: "wafer/us",
        class: Class::Standard,
        card: Card::new(tr("0.039", "4.8", "0.038", "0.039", None)),
    },
    OrEndpoint {
        host: "InferenceNet",
        tag: "inference-net/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.08", "4.4", "0.055", "0.08", None)),
    },
    OrEndpoint {
        host: "Reka",
        tag: "reka",
        class: Class::Standard,
        card: Card::new(tr("0.14", "4.2", "0.105", "0.14", None)),
    },
    OrEndpoint {
        host: "Makora",
        tag: "makora/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.14", "4.4", "0.1", "0.14", None)),
    },
    OrEndpoint {
        host: "Wafer",
        tag: "wafer",
        class: Class::Standard,
        card: Card::new(tr("0.17", "4.8", "0.16", "0.17", None)),
    },
    OrEndpoint {
        host: "AkashML",
        tag: "akashml/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.19", "4.4", "0.19", "0.19", None)),
    },
    OrEndpoint {
        host: "Sail Research",
        tag: "sail-research/us",
        class: Class::Standard,
        card: Card::new(tr("0.2", "3.4", "0.15", "0.2", None)),
    },
    OrEndpoint {
        host: "Sail Research",
        tag: "sail-research/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.2", "3.4", "0.15", "0.2", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.5625", "2.5", "0.125", "0.5625", None)),
    },
    OrEndpoint {
        host: "Inceptron",
        tag: "inceptron/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.6", "2.2", "0.2", "0.6", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.7", "2.2", "0.13", "0.7", None)),
    },
    OrEndpoint {
        host: "Io Net",
        tag: "io-net/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.75", "3.4", "0.2", "0.75", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala",
        class: Class::Standard,
        card: Card::new(tr("0.84", "2.64", "0.156", "0.84", None)),
    },
    OrEndpoint {
        host: "Decart",
        tag: "decart/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.8424", "2.6477", "0.1955", "0.8424", None)),
    },
    OrEndpoint {
        host: "DigitalOcean",
        tag: "digitalocean",
        class: Class::Standard,
        card: Card::new(tr("0.91", "2.86", "0.169", "0.91", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.98", "3.08", "0.182", "0.98", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.12", "3.52", "0.208", "1.12", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba",
        class: Class::Standard,
        card: Card::new(tr("1.19", "3.74", "0.238", "1.19", None)),
    },
    OrEndpoint {
        host: "Friendli",
        tag: "friendli",
        class: Class::Standard,
        card: Card::new(tr("1.26", "3.96", "0.234", "1.26", None)),
    },
    OrEndpoint {
        host: "Mistral",
        tag: "mistral/zdr",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.14", "1.4", None)),
    },
    OrEndpoint {
        host: "Baidu",
        tag: "baidu/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp4",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.14", "1.4", None)),
    },
    OrEndpoint {
        host: "Mistral",
        tag: "mistral/nvfp4",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.14", "1.4", None)),
    },
    OrEndpoint {
        host: "Crusoe",
        tag: "crusoe/fp4",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "PrimeIntellect",
        tag: "primeintellect",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Modal",
        tag: "modal",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp4",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.14", "1.4", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Cloudflare",
        tag: "cloudflare",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "AtlasCloud",
        tag: "atlas-cloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Z.AI",
        tag: "z-ai/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
    },
    OrEndpoint {
        host: "Mistral",
        tag: "mistral/nvfp4",
        class: Class::Standard,
        card: Card::new(tr("1.54", "4.84", "0.154", "1.54", None)),
    },
    OrEndpoint {
        host: "Morph",
        tag: "morph/fp8",
        class: Class::Standard,
        card: Card::new(tr("1.615", "6", "0.108", "1.615", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks/fast",
        class: Class::Fast,
        card: Card::new(tr("2.1", "6.6", "0.39", "2.1", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fast",
        class: Class::Fast,
        card: Card::new(tr("2.1", "6.6", "0.21", "2.1", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fast",
        class: Class::Fast,
        card: Card::new(tr("2.1", "6.6", "0.21", "2.1", None)),
    },
    OrEndpoint {
        host: "Alibaba",
        tag: "alibaba/fast",
        class: Class::Fast,
        card: Card::new(tr("2.8", "8.8", "0.56", "2.8", None)),
    },
];

/// `z-ai/glm-5.3-flash` (`GET /api/v1/models/z-ai/glm-5.3-flash/endpoints`).
static OR_Z_AI_GLM_5_3_FLASH: &[OrEndpoint] = &[
    OrEndpoint {
        host: "Relace",
        tag: "relace",
        class: Class::Standard,
        card: Card::new(tr("0.04", "0.5", "0.0125", "0.04", None)),
    },
    OrEndpoint {
        host: "OpenInference",
        tag: "open-inference/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.044", "0.45", "0.01", "0.044", None)),
    },
    OrEndpoint {
        host: "Sail Research",
        tag: "sail-research/us",
        class: Class::Standard,
        card: Card::new(tr("0.045", "0.6", "0.0285", "0.045", None)),
    },
    OrEndpoint {
        host: "Sail Research",
        tag: "sail-research/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.045", "0.6", "0.0285", "0.045", None)),
    },
    OrEndpoint {
        host: "Reka",
        tag: "reka",
        class: Class::Standard,
        card: Card::new(tr("0.06", "1.6", "0.04", "0.06", None)),
    },
    OrEndpoint {
        host: "StreamLake",
        tag: "streamlake/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.069", "0.23", "0.0138", "0.069", None)),
    },
    OrEndpoint {
        host: "InferenceNet",
        tag: "inference-net/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.07", "0.5", "0.036", "0.07", None)),
    },
    OrEndpoint {
        host: "Wafer",
        tag: "wafer",
        class: Class::Standard,
        card: Card::new(tr("0.07", "0.5", "0.03", "0.07", None)),
    },
    OrEndpoint {
        host: "DeepInfra",
        tag: "deepinfra/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.075", "0.25", "0.015", "0.075", None)),
    },
    OrEndpoint {
        host: "Novita",
        tag: "novita/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.084", "0.28", "0.0168", "0.084", None)),
    },
    OrEndpoint {
        host: "GMICloud",
        tag: "gmicloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.09", "0.3", "0.018", "0.09", None)),
    },
    OrEndpoint {
        host: "Decart",
        tag: "decart/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.093", "0.31", "0.0186", "0.093", None)),
    },
    OrEndpoint {
        host: "DekaLLM",
        tag: "dekallm",
        class: Class::Standard,
        card: Card::new(tr("0.1", "1", "0.04", "0.1", None)),
    },
    OrEndpoint {
        host: "Near AI",
        tag: "near-ai/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.105", "0.35", "0.0245", "0.105", None)),
    },
    OrEndpoint {
        host: "Morph",
        tag: "morph/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.111", "0.75", "0.015", "0.111", None)),
    },
    OrEndpoint {
        host: "Phala",
        tag: "phala/nvfp4",
        class: Class::Standard,
        card: Card::new(tr("0.1125", "0.375", "0.0225", "0.1125", None)),
    },
    OrEndpoint {
        host: "Inceptron",
        tag: "inceptron/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.12", "0.55", "0.099", "0.12", None)),
    },
    OrEndpoint {
        host: "Modal",
        tag: "modal/nvfp4",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "Crusoe",
        tag: "crusoe/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "CoreWeave",
        tag: "coreweave/nvfp4",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.05", "0.15", None)),
    },
    OrEndpoint {
        host: "AtlasCloud",
        tag: "atlas-cloud/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "Friendli",
        tag: "friendli",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "SiliconFlow",
        tag: "siliconflow/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "DigitalOcean",
        tag: "digitalocean",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "Together",
        tag: "together",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fp4",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "BaseTen",
        tag: "baseten/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "Venice",
        tag: "venice",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "Z.AI",
        tag: "z-ai/fp8",
        class: Class::Standard,
        card: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
    },
    OrEndpoint {
        host: "Parasail",
        tag: "parasail/fast",
        class: Class::Fast,
        card: Card::new(tr("0.1875", "0.625", "0.0375", "0.1875", None)),
    },
    OrEndpoint {
        host: "Fireworks",
        tag: "fireworks/us",
        class: Class::Standard,
        card: Card::new(tr("0.225", "0.75", "0.045", "0.225", None)),
    },
];

/// Every catalog row, sorted by `model` (the order of `MODEL_ROUTES`).
pub static ROW_RATES: &[RowRates] = &[
    RowRates {
        model: "claude-fable-5",
        customer: ANTHROPIC_CLAUDE_FABLE_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_FABLE_5),
            },
        ],
    },
    RowRates {
        model: "claude-fable-5-1",
        customer: ANTHROPIC_CLAUDE_FABLE_5_1,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_FABLE_5_1),
            },
        ],
    },
    RowRates {
        model: "claude-haiku-4-5",
        customer: ANTHROPIC_CLAUDE_HAIKU_4_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::Bedrock,
                card: CostCard::Own(&BEDROCK_CLAUDE_HAIKU_4_5),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_HAIKU_4_5),
            },
        ],
    },
    RowRates {
        model: "claude-opus-4-5",
        customer: ANTHROPIC_CLAUDE_OPUS_4_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_OPUS_4_5),
            },
        ],
    },
    RowRates {
        model: "claude-opus-4-6",
        customer: ANTHROPIC_CLAUDE_OPUS_4_6,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_OPUS_4_6),
            },
        ],
    },
    RowRates {
        model: "claude-opus-4-7",
        customer: ANTHROPIC_CLAUDE_OPUS_4_7,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_OPUS_4_7),
            },
        ],
    },
    RowRates {
        model: "claude-opus-4-8",
        customer: ANTHROPIC_CLAUDE_OPUS_4_8,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::Bedrock,
                card: CostCard::Own(&BEDROCK_CLAUDE_OPUS_4_8),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_OPUS_4_8),
            },
        ],
    },
    RowRates {
        model: "claude-opus-5",
        customer: ANTHROPIC_CLAUDE_OPUS_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_OPUS_5),
            },
        ],
    },
    RowRates {
        model: "claude-opus-5-5",
        customer: ANTHROPIC_CLAUDE_OPUS_5_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_OPUS_5_5),
            },
        ],
    },
    RowRates {
        model: "claude-sonnet-4-5",
        customer: ANTHROPIC_CLAUDE_SONNET_4_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_SONNET_4_5),
            },
        ],
    },
    RowRates {
        model: "claude-sonnet-4-6",
        customer: ANTHROPIC_CLAUDE_SONNET_4_6,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_SONNET_4_6),
            },
        ],
    },
    RowRates {
        model: "claude-sonnet-5",
        customer: ANTHROPIC_CLAUDE_SONNET_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_SONNET_5),
            },
        ],
    },
    RowRates {
        model: "claude-sonnet-5-5",
        customer: ANTHROPIC_CLAUDE_SONNET_5_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::Anthropic,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_ANTHROPIC_CLAUDE_SONNET_5_5),
            },
        ],
    },
    RowRates {
        model: "deepseek-flash",
        customer: DEEPSEEK_DEEPSEEK_FLASH,
        cost: &[
            CandidateCost {
                provider: ProviderId::DeepSeek,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_DEEPSEEK_AI_DEEPSEEK_V4_1_FLASH),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_DEEPSEEK_DEEPSEEK_V4_1_FLASH),
            },
        ],
    },
    RowRates {
        model: "deepseek-v4-pro",
        customer: DEEPSEEK_DEEPSEEK_V4_PRO,
        cost: &[
            CandidateCost {
                provider: ProviderId::DeepSeek,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_DEEPSEEK_AI_DEEPSEEK_V4_PRO_0813),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_DEEPSEEK_DEEPSEEK_V4_PRO_0813),
            },
        ],
    },
    RowRates {
        model: "google/gemma-4-31b-it",
        customer: Card::new(tr("0.09", "0.34", "0.05", "0.09", None)),
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_GOOGLE_GEMMA_4_31B_IT),
        }],
    },
    RowRates {
        model: "gpt-4.1",
        customer: OPENAI_GPT_4_1,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_4_1),
            },
        ],
    },
    RowRates {
        model: "gpt-4.1-mini",
        customer: OPENAI_GPT_4_1_MINI,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_4_1_MINI),
            },
        ],
    },
    RowRates {
        model: "gpt-4o",
        customer: OPENAI_GPT_4O,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_4O),
            },
        ],
    },
    RowRates {
        model: "gpt-4o-mini",
        customer: OPENAI_GPT_4O_MINI,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_4O_MINI),
            },
        ],
    },
    RowRates {
        model: "gpt-5",
        customer: OPENAI_GPT_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5),
            },
        ],
    },
    RowRates {
        model: "gpt-5-mini",
        customer: OPENAI_GPT_5_MINI,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_MINI),
            },
        ],
    },
    RowRates {
        model: "gpt-5-nano",
        customer: OPENAI_GPT_5_NANO,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_NANO),
            },
        ],
    },
    RowRates {
        model: "gpt-5-pro",
        customer: OPENAI_GPT_5_PRO,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_PRO),
            },
        ],
    },
    RowRates {
        model: "gpt-5.1",
        customer: OPENAI_GPT_5_1,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_1),
            },
        ],
    },
    RowRates {
        model: "gpt-5.2",
        customer: OPENAI_GPT_5_2,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_2),
            },
        ],
    },
    RowRates {
        model: "gpt-5.2-pro",
        customer: OPENAI_GPT_5_2_PRO,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_2_PRO),
            },
        ],
    },
    RowRates {
        model: "gpt-5.3-codex",
        customer: OPENAI_GPT_5_3_CODEX,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_3_CODEX),
            },
        ],
    },
    RowRates {
        model: "gpt-5.4",
        customer: OPENAI_GPT_5_4,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_4),
            },
        ],
    },
    RowRates {
        model: "gpt-5.4-mini",
        customer: OPENAI_GPT_5_4_MINI,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_4_MINI),
            },
        ],
    },
    RowRates {
        model: "gpt-5.4-nano",
        customer: OPENAI_GPT_5_4_NANO,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_4_NANO),
            },
        ],
    },
    RowRates {
        model: "gpt-5.4-pro",
        customer: OPENAI_GPT_5_4_PRO,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_4_PRO),
            },
        ],
    },
    RowRates {
        model: "gpt-5.5",
        customer: OPENAI_GPT_5_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_5),
            },
        ],
    },
    RowRates {
        model: "gpt-5.5-pro",
        customer: OPENAI_GPT_5_5_PRO,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_5_PRO),
            },
        ],
    },
    RowRates {
        model: "gpt-5.6-luna",
        customer: OPENAI_GPT_5_6_LUNA,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_6_LUNA),
            },
        ],
    },
    RowRates {
        model: "gpt-5.6-luna-pro",
        customer: OPENAI_GPT_5_6_LUNA,
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_OPENAI_GPT_5_6_LUNA_PRO),
        }],
    },
    RowRates {
        model: "gpt-5.6-sol",
        customer: OPENAI_GPT_5_6_SOL,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_6_SOL),
            },
        ],
    },
    RowRates {
        model: "gpt-5.6-sol-pro",
        customer: OPENAI_GPT_5_6_SOL,
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_OPENAI_GPT_5_6_SOL_PRO),
        }],
    },
    RowRates {
        model: "gpt-5.6-terra",
        customer: OPENAI_GPT_5_6_TERRA,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_5_6_TERRA),
            },
        ],
    },
    RowRates {
        model: "gpt-5.6-terra-pro",
        customer: OPENAI_GPT_5_6_TERRA,
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_OPENAI_GPT_5_6_TERRA_PRO),
        }],
    },
    RowRates {
        model: "gpt-6-astra",
        customer: OPENAI_GPT_6_ASTRA,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_6_ASTRA),
            },
        ],
    },
    RowRates {
        model: "gpt-6-astra-pro",
        customer: OPENAI_GPT_6_ASTRA,
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_OPENAI_GPT_6_ASTRA_PRO),
        }],
    },
    RowRates {
        model: "gpt-6-luna",
        customer: OPENAI_GPT_6_LUNA,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_6_LUNA),
            },
        ],
    },
    RowRates {
        model: "gpt-6-luna-pro",
        customer: OPENAI_GPT_6_LUNA,
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_OPENAI_GPT_6_LUNA_PRO),
        }],
    },
    RowRates {
        model: "gpt-6-sol",
        customer: OPENAI_GPT_6_SOL,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_6_SOL),
            },
        ],
    },
    RowRates {
        model: "gpt-6-sol-pro",
        customer: OPENAI_GPT_6_SOL,
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_OPENAI_GPT_6_SOL_PRO),
        }],
    },
    RowRates {
        model: "gpt-6.1-sol",
        customer: OPENAI_GPT_6_1_SOL,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_6_1_SOL),
            },
        ],
    },
    RowRates {
        model: "gpt-6.1-sol-pro",
        customer: OPENAI_GPT_6_1_SOL,
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_OPENAI_GPT_6_1_SOL_PRO),
        }],
    },
    RowRates {
        model: "grok-4.20",
        customer: XAI_GROK_4_20,
        cost: &[
            CandidateCost {
                provider: ProviderId::XAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_X_AI_GROK_4_20),
            },
        ],
    },
    RowRates {
        model: "grok-4.20-multi-agent",
        customer: XAI_GROK_4_20_MULTI_AGENT,
        cost: &[
            CandidateCost {
                provider: ProviderId::XAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_X_AI_GROK_4_20_MULTI_AGENT),
            },
        ],
    },
    RowRates {
        model: "grok-4.3",
        customer: XAI_GROK_4_3,
        cost: &[
            CandidateCost {
                provider: ProviderId::XAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_X_AI_GROK_4_3),
            },
        ],
    },
    RowRates {
        model: "grok-4.5",
        customer: XAI_GROK_4_5,
        cost: &[
            CandidateCost {
                provider: ProviderId::XAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_X_AI_GROK_4_5),
            },
        ],
    },
    RowRates {
        model: "grok-4.6",
        customer: XAI_GROK_4_6,
        cost: &[
            CandidateCost {
                provider: ProviderId::XAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_X_AI_GROK_4_6),
            },
        ],
    },
    RowRates {
        model: "grok-4.7",
        customer: XAI_GROK_4_7,
        cost: &[
            CandidateCost {
                provider: ProviderId::XAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_X_AI_GROK_4_7),
            },
        ],
    },
    RowRates {
        model: "grok-build-0.1",
        customer: XAI_GROK_BUILD_0_1,
        cost: &[
            CandidateCost {
                provider: ProviderId::XAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_X_AI_GROK_BUILD_0_1),
            },
        ],
    },
    RowRates {
        model: "llama-3.1-8b-instant",
        customer: Card::new(tr("0.05", "0.08", "0.025", "0.05", None)),
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_META_LLAMA_LLAMA_3_1_8B_INSTRUCT),
        }],
    },
    RowRates {
        model: "llama-3.3-70b-versatile",
        customer: Card::new(tr("1.04", "1.04", "1.04", "1.04", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_META_LLAMA_LLAMA_3_3_70B_INSTRUCT_TURBO),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_META_LLAMA_LLAMA_3_3_70B_INSTRUCT),
            },
        ],
    },
    RowRates {
        model: "meta-llama/llama-4-maverick",
        customer: Card::new(tr("0.1875", "0.6525", "0.1875", "0.1875", None)),
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_META_LLAMA_LLAMA_4_MAVERICK),
        }],
    },
    RowRates {
        model: "meta-llama/llama-4-scout",
        customer: Card::new(tr("0.1", "0.3", "0.1", "0.1", None)),
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_META_LLAMA_LLAMA_4_SCOUT),
        }],
    },
    RowRates {
        model: "meta/muse-glimmer-30b",
        customer: Card::new(tr("0.35", "1.5", "0.04", "0.35", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_META_MODELS_MUSE_GLIMMER_30B),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_META_MUSE_GLIMMER_30B),
            },
        ],
    },
    RowRates {
        model: "minimax/minimax-m3",
        customer: Card::new(tr("0.3", "1.2", "0.06", "0.3", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_MINIMAXAI_MINIMAX_M3),
            },
            CandidateCost {
                provider: ProviderId::Fireworks,
                card: CostCard::Own(&FIREWORKS_ACCOUNTS_FIREWORKS_MODELS_MINIMAX_M3),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_MINIMAX_MINIMAX_M3),
            },
        ],
    },
    RowRates {
        model: "moonshotai/kimi-k2.6",
        customer: Card::new(tr("0.95", "4", "0.16", "0.95", None)),
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_MOONSHOTAI_KIMI_K2_6),
        }],
    },
    RowRates {
        model: "moonshotai/kimi-k2.7-code",
        customer: Card::new(tr("0.95", "4", "0.19", "0.95", None)),
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_MOONSHOTAI_KIMI_K2_7_CODE),
        }],
    },
    RowRates {
        model: "moonshotai/kimi-k3",
        customer: Card::new(tr("3", "15", "0.3", "3", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_MOONSHOTAI_KIMI_K3),
            },
            CandidateCost {
                provider: ProviderId::Fireworks,
                card: CostCard::Own(&FIREWORKS_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_MOONSHOTAI_KIMI_K3),
            },
        ],
    },
    RowRates {
        model: "o3",
        customer: OPENAI_O3,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_O3),
            },
        ],
    },
    RowRates {
        model: "o3-pro",
        customer: OPENAI_O3_PRO,
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_OPENAI_O3_PRO),
        }],
    },
    RowRates {
        model: "openai/gpt-oss-120b",
        customer: GROQ_OPENAI_GPT_OSS_120B,
        cost: &[
            CandidateCost {
                provider: ProviderId::Groq,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_OPENAI_GPT_OSS_120B),
            },
            CandidateCost {
                provider: ProviderId::Fireworks,
                card: CostCard::Own(&FIREWORKS_ACCOUNTS_FIREWORKS_MODELS_GPT_OSS_120B),
            },
        ],
    },
    RowRates {
        model: "openai/gpt-oss-20b",
        customer: GROQ_OPENAI_GPT_OSS_20B,
        cost: &[
            CandidateCost {
                provider: ProviderId::Groq,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_OSS_20B),
            },
        ],
    },
    RowRates {
        model: "openai/gpt-oss-safeguard-20b",
        customer: GROQ_OPENAI_GPT_OSS_SAFEGUARD_20B,
        cost: &[
            CandidateCost {
                provider: ProviderId::Groq,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_GPT_OSS_SAFEGUARD_20B),
            },
        ],
    },
    RowRates {
        model: "qwen/qwen-2.5-7b-instruct",
        customer: Card::new(tr("0.1", "0.2", "0.1", "0.1", None)),
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_QWEN_QWEN_2_5_7B_INSTRUCT),
        }],
    },
    RowRates {
        model: "qwen/qwen3.5-9b",
        customer: Card::new(tr("0.17", "0.25", "0.17", "0.17", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_QWEN_QWEN3_5_9B),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_QWEN_QWEN3_5_9B),
            },
        ],
    },
    RowRates {
        model: "qwen/qwen3.6-plus",
        customer: Card::new(tr("0.5", "3", "0.5", "0.5", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_QWEN_QWEN3_6_PLUS),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_QWEN_QWEN3_6_PLUS),
            },
        ],
    },
    RowRates {
        model: "qwen/qwen3.7-max",
        customer: Card::new(tr("2.5", "7.5", "0.5", "2.5", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_QWEN_QWEN3_7_MAX),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_QWEN_QWEN3_7_MAX),
            },
        ],
    },
    RowRates {
        model: "qwen/qwen3.7-plus",
        customer: Card::new(tr("0.32", "1.28", "0.32", "0.32", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_QWEN_QWEN3_7_PLUS),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_QWEN_QWEN3_7_PLUS),
            },
        ],
    },
    RowRates {
        model: "qwen/qwen3.8-2.4t-a95b",
        customer: Card::new(tr("2", "6", "0.25", "2", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_QWEN_QWEN3_8_2_4T_A95B),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_QWEN_QWEN3_8_2_4T_A95B),
            },
        ],
    },
    RowRates {
        model: "qwen/qwen3.8-27b",
        customer: GROQ_QWEN_QWEN3_8_27B,
        cost: &[
            CandidateCost {
                provider: ProviderId::Groq,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_QWEN_QWEN3_8_27B),
            },
        ],
    },
    RowRates {
        model: "qwen/qwen3.8-flash",
        customer: Card::new(tr("0.15", "0.47", "0.15", "0.15", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_QWEN_QWEN3_8_FLASH),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_QWEN_QWEN3_8_FLASH),
            },
        ],
    },
    RowRates {
        model: "text-embedding-3-large",
        customer: OPENAI_TEXT_EMBEDDING_3_LARGE,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_TEXT_EMBEDDING_3_LARGE),
            },
        ],
    },
    RowRates {
        model: "text-embedding-3-small",
        customer: OPENAI_TEXT_EMBEDDING_3_SMALL,
        cost: &[
            CandidateCost {
                provider: ProviderId::OpenAi,
                card: CostCard::Customer,
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_OPENAI_TEXT_EMBEDDING_3_SMALL),
            },
        ],
    },
    RowRates {
        model: "thinkingmachines/inkling",
        customer: Card::new(tr("1", "4.05", "0.17", "1", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_THINKINGMACHINES_INKLING),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_THINKINGMACHINES_INKLING),
            },
        ],
    },
    RowRates {
        model: "z-ai/glm-5.1",
        customer: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
        cost: &[CandidateCost {
            provider: ProviderId::OpenRouter,
            card: CostCard::OpenRouter(OR_Z_AI_GLM_5_1),
        }],
    },
    RowRates {
        model: "z-ai/glm-5.2",
        customer: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_ZAI_ORG_GLM_5_2),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_Z_AI_GLM_5_2),
            },
        ],
    },
    RowRates {
        model: "z-ai/glm-5.3",
        customer: Card::new(tr("1.4", "4.4", "0.26", "1.4", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_ZAI_ORG_GLM_5_3),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_Z_AI_GLM_5_3),
            },
        ],
    },
    RowRates {
        model: "z-ai/glm-5.3-flash",
        customer: Card::new(tr("0.15", "0.5", "0.03", "0.15", None)),
        cost: &[
            CandidateCost {
                provider: ProviderId::Together,
                card: CostCard::Own(&TOGETHER_ZAI_ORG_GLM_5_3_FLASH),
            },
            CandidateCost {
                provider: ProviderId::OpenRouter,
                card: CostCard::OpenRouter(OR_Z_AI_GLM_5_3_FLASH),
            },
        ],
    },
];
