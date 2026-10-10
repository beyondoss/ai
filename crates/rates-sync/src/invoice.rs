//! Invoice-level truth: what each vendor actually billed the organization, from its admin cost
//! API, set against the table. For every (day, model, token kind) the vendor reports both an
//! amount and a token count for, `tokens × rate` must be the amount. A disagreement means the
//! card is not what the vendor bills.
//!
//! The reports are the organization's own spend: they are read live and never snapshotted or
//! committed. Needs `OPENAI_ADMIN_KEY` / `ANTHROPIC_ADMIN_KEY`; a vendor without its key is
//! skipped and said so.

use crate::Result;
use crate::build::{Card, Table};
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

/// OpenAI: `/v1/organization/costs` by line item, `/v1/organization/usage/completions` by model,
/// daily, over the last `days` days.
pub fn openai(table: &Table, key: &str, days: u64, now: u64) -> Result<Vec<Check>> {
    let c = client()?;
    let start = (now / 86_400 - days) * 86_400;
    let auth = [("authorization", format!("Bearer {key}"))];
    let mut costs: BTreeMap<(i64, String, String), f64> = BTreeMap::new();
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
                let Some((model, kind)) = li.split_once(", ") else {
                    continue;
                };
                let amt = r
                    .pointer("/amount/value")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                *costs
                    .entry((day, model.to_owned(), kind.to_owned()))
                    .or_default() += amt;
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
    let mut usage: BTreeMap<(i64, String), (u64, u64, u64)> = BTreeMap::new();
    let mut page: Option<String> = None;
    loop {
        let mut url = format!(
            "https://api.openai.com/v1/organization/usage/completions?start_time={start}&group_by=model&group_by=batch&group_by=service_tier&bucket_width=1d&limit=31"
        );
        if let Some(p) = &page {
            url.push_str(&format!("&page={p}"));
        }
        let v: Value = serde_json::from_slice(&get(&c, &url, &auth)?)
            .map_err(|e| format!("OpenAI usage: {e}"))?;
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
                let Some(model) = r.get("model").and_then(Value::as_str) else {
                    continue;
                };
                let batch = r.get("batch").and_then(Value::as_bool).unwrap_or(false);
                let tier = r
                    .get("service_tier")
                    .and_then(Value::as_str)
                    .unwrap_or("default");
                // Only standard-tier, non-batch traffic is compared with the standard card.
                if batch || !(tier == "default" || tier == "standard" || tier == "auto") {
                    continue;
                }
                let n = |k: &str| r.get(k).and_then(Value::as_u64).unwrap_or(0);
                let e = usage.entry((day, model.to_owned())).or_default();
                e.0 += n("input_tokens");
                e.1 += n("input_cached_tokens");
                e.2 += n("output_tokens");
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
    let mut out = Vec::new();
    for ((day, model), (input, cached, output)) in usage {
        let Some((card_name, card)) = card_for(table, "openai", &model) else {
            continue;
        };
        let s = &card.standard;
        for (kind, field, tokens, rate) in [
            ("input", "input", input.saturating_sub(cached), s.input),
            ("cached input", "cache_read", cached, s.cache_read),
            ("output", "output", output, s.output),
        ] {
            if tokens == 0 {
                continue;
            }
            let Some(billed) = costs.get(&(day, model.clone(), kind.to_owned())).copied() else {
                continue;
            };
            #[allow(clippy::cast_precision_loss)]
            let expected = tokens as f64 * usd(rate) / 1e6;
            out.push(Check {
                vendor: "openai",
                card: card_name.to_owned(),
                field,
                day: crate::fetch::date_of(day),
                model: model.clone(),
                kind: kind.to_owned(),
                tokens,
                billed,
                expected,
                ok: close(billed, expected),
                superseded: false,
            });
        }
    }
    Ok(out)
}

/// Anthropic: `/v1/organizations/cost_report` grouped by description (model, token type, tier,
/// geo, context window) against `/v1/organizations/usage_report/messages` grouped the same way.
pub fn anthropic(table: &Table, key: &str, days: u64, now: u64) -> Result<Vec<Check>> {
    let c = client()?;
    let start = crate::fetch::date_of(i64::try_from((now / 86_400 - days) * 86_400).unwrap_or(0));
    let end = crate::fetch::date_of(i64::try_from(now / 86_400 * 86_400).unwrap_or(0));
    let auth = [
        ("x-api-key", key.to_owned()),
        ("anthropic-version", "2023-06-01".to_owned()),
    ];
    // (day, model, token_type) → cents, standard tier, global geo, standard speed only.
    let mut costs: BTreeMap<(String, String, String), f64> = BTreeMap::new();
    let mut page: Option<String> = None;
    loop {
        let mut url = format!(
            "https://api.anthropic.com/v1/organizations/cost_report?starting_at={start}T00:00:00Z&ending_at={end}T00:00:00Z&group_by[]=description&limit=31"
        );
        if let Some(p) = &page {
            url.push_str(&format!("&page={p}"));
        }
        let v: Value = serde_json::from_slice(&get(&c, &url, &auth)?)
            .map_err(|e| format!("Anthropic cost report: {e}"))?;
        for b in v
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let day = b
                .get("starting_at")
                .and_then(Value::as_str)
                .unwrap_or("")
                .chars()
                .take(10)
                .collect::<String>();
            for r in b
                .get("results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let g = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("");
                if g("cost_type") != "tokens"
                    || g("service_tier") != "standard"
                    || !(g("inference_geo") == "global" || g("inference_geo") == "not_available")
                    || r.get("speed")
                        .and_then(Value::as_str)
                        .is_some_and(|s| s != "standard")
                {
                    continue;
                }
                let amt: f64 = g("amount").parse().unwrap_or(0.0);
                *costs
                    .entry((
                        day.clone(),
                        g("model").to_owned(),
                        g("token_type").to_owned(),
                    ))
                    .or_default() += amt;
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
    let mut usage: BTreeMap<(String, String, String), u64> = BTreeMap::new();
    let mut page: Option<String> = None;
    loop {
        let mut url = format!(
            "https://api.anthropic.com/v1/organizations/usage_report/messages?starting_at={start}T00:00:00Z&ending_at={end}T00:00:00Z&group_by[]=model&group_by[]=service_tier&group_by[]=inference_geo&bucket_width=1d&limit=31"
        );
        if let Some(p) = &page {
            url.push_str(&format!("&page={p}"));
        }
        let v: Value = serde_json::from_slice(&get(&c, &url, &auth)?)
            .map_err(|e| format!("Anthropic usage report: {e}"))?;
        for b in v
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let day = b
                .get("starting_at")
                .and_then(Value::as_str)
                .unwrap_or("")
                .chars()
                .take(10)
                .collect::<String>();
            for r in b
                .get("results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let g = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("");
                if g("service_tier") != "standard"
                    || !(g("inference_geo") == "global"
                        || g("inference_geo") == "not_available"
                        || g("inference_geo").is_empty())
                    || !(g("speed").is_empty() || g("speed") == "standard")
                {
                    continue;
                }
                let model = g("model").to_owned();
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
                        .entry((day.clone(), model.clone(), tt.to_owned()))
                        .or_default() += n(p);
                }
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
    let mut out = Vec::new();
    for ((day, model, tt), tokens) in usage {
        if tokens == 0 {
            continue;
        }
        let Some((card_name, card)) = card_for(table, "anthropic", &model) else {
            continue;
        };
        let s = &card.standard;
        let (field, rate) = match tt.as_str() {
            "uncached_input_tokens" => ("input", s.input),
            "cache_read_input_tokens" => ("cache_read", s.cache_read),
            "cache_creation.ephemeral_5m_input_tokens" => ("cache_write", s.cache_write_5m),
            "cache_creation.ephemeral_1h_input_tokens" => match s.cache_write_1h {
                Some(r) => ("cache_write_1h", r),
                None => continue,
            },
            _ => ("output", s.output),
        };
        let Some(cents) = costs
            .get(&(day.clone(), model.clone(), tt.clone()))
            .copied()
        else {
            continue;
        };
        let billed = cents / 100.0;
        #[allow(clippy::cast_precision_loss)]
        let expected = tokens as f64 * usd(rate) / 1e6;
        out.push(Check {
            vendor: "anthropic",
            card: card_name.to_owned(),
            field,
            day,
            model,
            kind: tt,
            tokens,
            billed,
            expected,
            ok: close(billed, expected),
            superseded: false,
        });
    }
    Ok(out)
}
