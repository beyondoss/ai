//! Invoice-level truth: what each vendor actually billed the organization, from its admin cost
//! API, set against the table. For every (day, model, token kind, class, tier, residency) the
//! vendor reports both an amount and a token count for, `tokens × rate` must be the amount, at the
//! card's rate for that class (fast mode, flex), tier (long context) and residency (US-only). A
//! disagreement means the card is not what the vendor bills.
//!
//! The reports are the organization's own spend: they are read live and never snapshotted or
//! committed. Needs `OPENAI_ADMIN_KEY` / `ANTHROPIC_ADMIN_KEY`; a vendor without its key is
//! skipped and said so.

use crate::Result;
use crate::build::{Card, Table, Tr};
use crate::dec::Rate;
use crate::fetch::{client, get};
use serde_json::Value;
use std::collections::BTreeMap;

/// One comparison: what was billed, what the card says.
pub struct Check {
    pub vendor: &'static str,
    /// The card the line is checked against, and the card field (`input`, `cache_read`, …).
    pub card: String,
    pub field: &'static str,
    pub day: String,
    pub model: String,
    pub kind: String,
    pub tokens: u64,
    /// Billed, in USD (exact decimal text from the report).
    pub billed: f64,
    pub expected: f64,
    pub ok: bool,
    /// Agrees only with a rate the vendor has since changed (`[[superseded]]`).
    pub superseded: bool,
}

fn card_for<'a>(table: &'a Table, vendor: &str, model: &str) -> Option<(&'a str, &'a Card)> {
    // A billed model id may carry a date (`gpt-5-2025-08-07`, `claude-haiku-4-5-20251001`).
    let mut best: Option<(&str, &str, &Card)> = None;
    for (name, c) in &table.cards {
        let Some(m) = name.strip_prefix(vendor).and_then(|x| x.strip_prefix(':')) else {
            continue;
        };
        let dated = model.strip_prefix(m).is_some_and(|rest| {
            rest.is_empty()
                || (rest.starts_with('-')
                    && rest[1..].chars().all(|c| c.is_ascii_digit() || c == '-'))
        });
        if dated && best.is_none_or(|(b, _, _)| m.len() > b.len()) {
            best = Some((m, name.as_str(), c));
        }
    }
    best.map(|(_, n, c)| (n, c))
}

/// A line that disagrees with the current card but matches a rate the vendor has since changed
/// (a `[[superseded]]` entry, before its end date) is accepted.
pub fn accept_superseded(checks: &mut [Check], spec: &crate::spec::Spec) -> crate::Result<()> {
    for c in checks.iter_mut().filter(|c| !c.ok) {
        for s in &spec.superseded {
            if s.card == c.card && s.field == c.field && c.day.as_str() < s.until.as_str() {
                let old = Rate::per_million(&s.usd)?;
                #[allow(clippy::cast_precision_loss)]
                let expected = c.tokens as f64 * usd(old) / 1e6;
                if close(c.billed, expected) {
                    c.ok = true;
                    c.superseded = true;
                }
            }
        }
    }
    Ok(())
}

fn usd(r: Rate) -> f64 {
    // Display only: comparisons use a relative tolerance on the reported decimal.
    #[allow(clippy::cast_precision_loss)]
    let x = r.0 as f64 / 1e12;
    x
}

fn close(billed: f64, expected: f64) -> bool {
    // Reports round amounts (OpenAI to ~1e-9 USD, Anthropic to 1e-5 cents); allow that, no more.
    (billed - expected).abs() <= 1e-7_f64.max(expected * 1e-6)
}

/// The card's rate set for a class (`standard` / `fast` / `flex`) and tier, as the pricer picks it.
fn variant<'a>(card: &'a Card, class: &str, long: bool) -> Option<&'a Tr> {
    match (class, long) {
        ("standard", false) => Some(&card.standard),
        ("standard", true) => card.long.as_ref(),
        ("fast", false) => card.fast.as_ref(),
        ("fast", true) => card.fast_long.as_ref(),
        ("flex", false) => card.flex.as_ref(),
        ("flex", true) => card.flex_long.as_ref(),
        _ => None,
    }
}

/// `tokens × rate × multiplier` in USD, and the check against what was billed. A line the card has
/// no rate for is a failed check (expected `NaN`), never a skipped one.
#[allow(clippy::too_many_arguments)]
fn check(
    vendor: &'static str,
    card: &str,
    field: &'static str,
    day: String,
    model: &str,
    kind: String,
    tokens: u64,
    billed: f64,
    rate: Option<(Rate, u128)>,
) -> Check {
    #[allow(clippy::cast_precision_loss)]
    let expected = rate.map_or(f64::NAN, |(r, bps)| {
        tokens as f64 * usd(r) * (bps as f64 / 10_000.0) / 1e6
    });
    Check {
        vendor,
        card: card.to_owned(),
        field,
        day,
        model: model.to_owned(),
        kind,
        tokens,
        billed,
        expected,
        ok: close(billed, expected),
        superseded: false,
    }
}

/// One OpenAI cost line item, `[tier | ]model[, kind][, long context]`, as (card class, model,
/// card field, long context). `priority` is the card's fast class; an embeddings line names no
/// kind (input). An unknown tier or kind reads `unknown`, which no card prices.
fn openai_line(li: &str) -> (&'static str, &str, &'static str, bool) {
    let (tier, rest) = li.split_once(" | ").unwrap_or(("default", li));
    let (rest, long) = match rest.strip_suffix(", long context") {
        Some(r) => (r, true),
        None => (rest, false),
    };
    let (model, kind) = rest.split_once(", ").unwrap_or((rest, "input"));
    let class = match tier {
        "default" => "standard",
        "priority" => "fast",
        "flex" => "flex",
        _ => "unknown",
    };
    let field = match kind {
        "input" => "input",
        "cached input" => "cache_read",
        "cache writes" => "cache_write",
        "output" => "output",
        _ => "unknown",
    };
    (class, model, field, long)
}

/// OpenAI: `/v1/organization/costs` by line item, daily, over the last `days` days. Each line
/// carries its own `quantity` in tokens and names everything that priced it, so each is checked
/// on its own: `[tier | ]model[, kind][, long context]`, where the tier is `priority` (the card's
/// fast class) or `flex`, and the kind is `input`, `cached input`, `cache writes` or `output`
/// (an embeddings line names no kind: input).
pub fn openai(table: &Table, key: &str, days: u64, now: u64) -> Result<Vec<Check>> {
    let c = client()?;
    let start = (now / 86_400 - days) * 86_400;
    let auth = [("authorization", format!("Bearer {key}"))];
    let mut out = Vec::new();
    let mut page: Option<String> = None;
    loop {
        let mut url = format!(
            "https://api.openai.com/v1/organization/costs?start_time={start}&group_by=line_item&limit=180"
        );
        if let Some(p) = &page {
            url.push_str(&format!("&page={p}"));
        }
        let v: Value = serde_json::from_slice(&get(&c, &url, &auth)?)
            .map_err(|e| format!("OpenAI costs: {e}"))?;
        for b in v
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let day = b.get("start_time").and_then(Value::as_i64).unwrap_or(0);
            for r in b
                .get("results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let Some(li) = r.get("line_item").and_then(Value::as_str) else {
                    continue;
                };
                let billed = r
                    .pointer("/amount/value")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let tokens = r.get("quantity").and_then(Value::as_f64).unwrap_or(0.0) as u64;
                if tokens == 0 && billed == 0.0 {
                    continue;
                }
                let (class, model, field, long) = openai_line(li);
                // Tool calls and models the table does not carry are not card lines.
                let Some((card_name, card)) = card_for(table, "openai", model) else {
                    continue;
                };
                let rate = variant(card, class, long).and_then(|t| match field {
                    "input" => Some(t.input),
                    "cache_read" => Some(t.cache_read),
                    "cache_write" => Some(t.cache_write_5m),
                    "output" => Some(t.output),
                    _ => None,
                });
                out.push(check(
                    "openai",
                    card_name,
                    field,
                    crate::fetch::date_of(day),
                    model,
                    li.to_owned(),
                    tokens,
                    billed,
                    rate.map(|r| (r, 10_000)),
                ));
            }
        }
        page = v
            .get("next_page")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if !v.get("has_more").and_then(Value::as_bool).unwrap_or(false) || page.is_none() {
            break;
        }
    }
    Ok(out)
}

/// Anthropic: `/v1/organizations/cost_report` grouped by description against
/// `/v1/organizations/usage_report/messages` grouped by model, service tier, data residency and
/// speed (the fast mode beta header exposes it), per day and token type. A cost line is fast mode
/// when its description says ` - Fast - `, and US-only when its `inference_geo` is `us` (the card's
/// `geo_us` multiplier). A model priced by prompt length (Claude Haiku 5.5) is the one line the
/// reports cannot split: both bucket context windows as `0-200k`, coarser than its 100K threshold,
/// so its day line is checked to lie between its two tiers.
pub fn anthropic(table: &Table, key: &str, days: u64, now: u64) -> Result<Vec<Check>> {
    // (day, model, token_type, speed, inference_geo)
    type Key = (String, String, String, &'static str, String);
    let c = client()?;
    let start = crate::fetch::date_of(i64::try_from((now / 86_400 - days) * 86_400).unwrap_or(0));
    let end = crate::fetch::date_of(i64::try_from(now / 86_400 * 86_400).unwrap_or(0));
    let auth = [
        ("x-api-key", key.to_owned()),
        ("anthropic-version", "2023-06-01".to_owned()),
        ("anthropic-beta", "fast-mode-2026-02-01".to_owned()),
    ];
    let pages = |base: String, what: &str| -> Result<Vec<Value>> {
        let mut all = Vec::new();
        let mut page: Option<String> = None;
        loop {
            let mut url = base.clone();
            if let Some(p) = &page {
                url.push_str(&format!("&page={p}"));
            }
            let v: Value = serde_json::from_slice(&get(&c, &url, &auth)?)
                .map_err(|e| format!("Anthropic {what}: {e}"))?;
            page = v
                .get("next_page")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let more = v.get("has_more").and_then(Value::as_bool).unwrap_or(false);
            all.push(v);
            if !more || page.is_none() {
                return Ok(all);
            }
        }
    };
    let results = |pages: &[Value]| -> Vec<(String, Value)> {
        let mut out = Vec::new();
        for v in pages {
            for b in v
                .get("data")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let day: String = b
                    .get("starting_at")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .chars()
                    .take(10)
                    .collect();
                for r in b
                    .get("results")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    out.push((day.clone(), r.clone()));
                }
            }
        }
        out
    };
    let geo = |r: &Value| match r.get("inference_geo").and_then(Value::as_str) {
        Some("us") => "us".to_owned(),
        _ => "global".to_owned(),
    };

    // Cents per key. A non-standard service tier (Priority Tier, contract-priced) has no card rate:
    // it is kept as its own key and fails below.
    let mut costs: BTreeMap<Key, f64> = BTreeMap::new();
    let mut tiers: BTreeMap<Key, String> = BTreeMap::new();
    for (day, r) in results(&pages(
        format!(
            "https://api.anthropic.com/v1/organizations/cost_report?starting_at={start}T00:00:00Z&ending_at={end}T00:00:00Z&group_by[]=description&limit=31"
        ),
        "cost report",
    )?) {
        let g = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("");
        if g("cost_type") != "tokens" {
            continue;
        }
        let speed = if g("description").contains(" - Fast - ") {
            "fast"
        } else {
            "standard"
        };
        let k = (
            day,
            g("model").to_owned(),
            g("token_type").to_owned(),
            speed,
            geo(&r),
        );
        if g("service_tier") != "standard" {
            tiers.insert(k.clone(), g("service_tier").to_owned());
        }
        *costs.entry(k).or_default() += g("amount").parse::<f64>().unwrap_or(0.0);
    }
    let mut usage: BTreeMap<Key, u64> = BTreeMap::new();
    for (day, r) in results(&pages(
        format!(
            "https://api.anthropic.com/v1/organizations/usage_report/messages?starting_at={start}T00:00:00Z&ending_at={end}T00:00:00Z&group_by[]=model&group_by[]=service_tier&group_by[]=inference_geo&group_by[]=speed&bucket_width=1d&limit=31"
        ),
        "usage report",
    )?) {
        let g = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("");
        let speed = if g("speed") == "fast" {
            "fast"
        } else {
            "standard"
        };
        let n = |p: &str| r.pointer(p).and_then(Value::as_u64).unwrap_or(0);
        for (tt, p) in [
            ("uncached_input_tokens", "/uncached_input_tokens"),
            ("cache_read_input_tokens", "/cache_read_input_tokens"),
            (
                "cache_creation.ephemeral_5m_input_tokens",
                "/cache_creation/ephemeral_5m_input_tokens",
            ),
            (
                "cache_creation.ephemeral_1h_input_tokens",
                "/cache_creation/ephemeral_1h_input_tokens",
            ),
            ("output_tokens", "/output_tokens"),
        ] {
            *usage
                .entry((
                    day.clone(),
                    g("model").to_owned(),
                    tt.to_owned(),
                    speed,
                    geo(&r),
                ))
                .or_default() += n(p);
        }
    }
    let mut out = Vec::new();
    for (k, cents) in costs {
        let tokens = usage.get(&k).copied().unwrap_or(0);
        let (day, model, tt, speed, g) = k.clone();
        let Some((card_name, card)) = card_for(table, "anthropic", &model) else {
            continue;
        };
        let field = match tt.as_str() {
            "uncached_input_tokens" => "input",
            "cache_read_input_tokens" => "cache_read",
            "cache_creation.ephemeral_5m_input_tokens" => "cache_write",
            "cache_creation.ephemeral_1h_input_tokens" => "cache_write_1h",
            "output_tokens" => "output",
            _ => "unknown",
        };
        let rate_of = |t: &Tr| match field {
            "input" => Some(t.input),
            "cache_read" => Some(t.cache_read),
            "cache_write" => Some(t.cache_write_5m),
            "cache_write_1h" => t.cache_write_1h,
            "output" => Some(t.output),
            _ => None,
        };
        let mult = if g == "us" { card.geo_us } else { Some(10_000) };
        let rate = if tiers.contains_key(&k) {
            None
        } else {
            variant(card, speed, false).and_then(rate_of).zip(mult)
        };
        let kind = format!("{tt} ({speed}, {g})");
        let mut chk = check(
            "anthropic",
            card_name,
            field,
            day,
            &model,
            kind,
            tokens,
            cents / 100.0,
            rate,
        );
        // Prompt-length pricing: the day's line mixes the model's two tiers (see above).
        if let (false, Some(long), Some(m)) =
            (chk.ok, variant(card, speed, true).and_then(rate_of), mult)
        {
            #[allow(clippy::cast_precision_loss)]
            let most = tokens as f64 * usd(long) * (m as f64 / 10_000.0) / 1e6;
            chk.ok = close(chk.billed, most) || (chk.expected..=most).contains(&chk.billed);
        }
        out.push(chk);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::openai_line;

    /// Every line-item shape OpenAI's cost report used over 30 days (read 2026-10-11).
    #[test]
    fn openai_line_items_name_their_class_tier_and_kind() {
        for (li, want) in [
            ("gpt-5.5, input", ("standard", "gpt-5.5", "input", false)),
            (
                "gpt-6-luna, cache writes, long context",
                ("standard", "gpt-6-luna", "cache_write", true),
            ),
            (
                "gpt-6-luna, output, long context",
                ("standard", "gpt-6-luna", "output", true),
            ),
            (
                "priority | gpt-5-mini-2025-08-07, cached input",
                ("fast", "gpt-5-mini-2025-08-07", "cache_read", false),
            ),
            (
                "flex | gpt-5-nano-2025-08-07, output",
                ("flex", "gpt-5-nano-2025-08-07", "output", false),
            ),
            (
                "gpt-6-sol, cache writes - 12hr",
                ("standard", "gpt-6-sol", "unknown", false),
            ),
            (
                "text-embedding-3-large",
                ("standard", "text-embedding-3-large", "input", false),
            ),
            (
                "scale | gpt-5.5, input",
                ("unknown", "gpt-5.5", "input", false),
            ),
        ] {
            assert_eq!(openai_line(li), want, "{li}");
        }
    }
}
