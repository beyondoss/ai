//! The rate tables behind [`crate::pricing`]: per catalog row, its list card (the primary
//! vendor's published card) and what each candidate costs us. Pricing is pass-through: the
//! customer pays what the serving candidate costs us, so the list card prices a row only when its
//! own vendor served it.
//!
//! The table itself, [`ROW_RATES`] and [`OPENROUTER_CREDIT_FEE`], is generated: `generated.rs`
//! is written by `mise run rates:sync` (crates/rates-sync) from the vendors' primary sources,
//! snapshotted under `verify/rates_sources/`, and the prose-only rules in
//! `verify/catalog_truth.toml`, each with its cited quote. Nobody edits it by hand;
//! `rates_generated_from_sources` regenerates it from the snapshots and fails on any difference.
//! A list card's standard tier is the catalog's [`crate::ListPrice`] for the row
//! (`list_card_standard_is_the_list_price`), so `GET /v1/models` publishes exactly what the
//! primary vendor charges.
//!
//! Changing any rate changes [`RATE_VERSION`], which every priced row logs: a row can always be
//! traced to the exact table that priced it, and repriced against another.
//!
//! # Maintenance
//!
//! Run `mise run rates:sync`: it re-fetches every source, regenerates the table, and prints what
//! moved, and sets the new [`RATE_VERSION`] (`rates-sync rate-version`, from [`table_hash`]). A
//! vendor that changes a page's layout fails the sync loudly (its parser never guesses); a vendor
//! that rewords a prose rule fails its quote check. The daily `rates-drift` workflow runs the same
//! fetch and opens a PR with what moved, or an issue when a source no longer reads.

use crate::ProviderId;
use crate::pricing::{Card, Class, TokenRates, usd};

// The generated table. `rustfmt::skip`: the generator lays it out, and a rustfmt release must
// never make the committed file differ from what the snapshots generate.
#[rustfmt::skip]
mod generated;
pub use generated::{OPENROUTER_CREDIT_FEE, ROW_RATES};

/// The exact rate table compiled into this build: the date its rates were checked, and a hash of
/// the table (`rate_version_names_this_table` recomputes it, and fails with the new value when
/// any rate changes). Logged on every priced `ai.usage` row.
pub const RATE_VERSION: &str = "2026-10-10.7e6c2a4eff15a788";

/// What one catalog row costs and charges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRates {
    /// The catalog row's name.
    pub model: &'static str,
    /// The primary vendor's published card. Its standard tier is the row's
    /// [`crate::ListPrice`]; it prices the rows that vendor serves ([`CostCard::List`]).
    pub list: Card,
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
    /// The list card: the candidate is the row's own vendor.
    List,
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

/// The hash half of [`RATE_VERSION`]: FNV-1a 64 over the compiled table's `Debug` text and the
/// OpenRouter fee. `rates-sync rate-version` writes it; `rate_version_names_this_table` checks it.
pub fn table_hash() -> String {
    let text = format!("{ROW_RATES:?}{OPENROUTER_CREDIT_FEE}");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
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
