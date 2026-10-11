//! A canonical one-line-per-card text of the table compiled into this build
//! (`rates-sync dump`), for reviewing a regeneration: two dumps diff to exactly the rates that
//! moved.

use providers::pricing::{Card, TokenRates};
use providers::rates::{CostCard, ROW_RATES};
use std::fmt::Write as _;

fn usd(micros: u64) -> String {
    let whole = micros / 1_000_000;
    let mut frac = format!("{:06}", micros % 1_000_000);
    while frac.ends_with('0') {
        frac.pop();
    }
    if frac.is_empty() {
        whole.to_string()
    } else {
        format!("{whole}.{frac}")
    }
}

fn rates(r: &TokenRates) -> String {
    format!(
        "in {} out {} read {} write {} write1h {}",
        usd(r.input),
        usd(r.output),
        usd(r.cache_read),
        usd(r.cache_write_5m),
        r.cache_write_1h.map_or("-".into(), usd)
    )
}

pub fn card(c: &Card) -> String {
    let mut out = format!("standard[{}]", rates(&c.standard));
    for (k, v) in [
        ("long", c.long),
        ("fast", c.fast),
        ("fast_long", c.fast_long),
        ("ultrafast", c.ultrafast),
        ("ultrafast_long", c.ultrafast_long),
        ("flex", c.flex),
        ("flex_long", c.flex_long),
    ] {
        if let Some(r) = v {
            let _ = write!(out, " {k}[{}]", rates(&r));
        }
    }
    if let Some(lc) = c.long_context {
        let _ = write!(
            out,
            " long_context {}{}",
            if lc.inclusive { ">=" } else { ">" },
            lc.threshold
        );
    }
    if let Some(g) = c.geo_us {
        let _ = write!(out, " geo_us {g}bps");
    }
    if !c.tools.is_empty() {
        let mut t: Vec<String> = c
            .tools
            .iter()
            .map(|(k, f)| format!("{}={}", k.as_str(), usd(*f)))
            .collect();
        t.sort();
        let _ = write!(out, " tools[{}]", t.join(","));
    }
    if let Some(op) = c.off_peak {
        let _ = write!(
            out,
            " off_peak[{:?} weekdays_only={} holidays={:?} covered={}..{} {}]",
            op.peak,
            op.weekdays_only,
            op.holidays,
            op.covered_from,
            op.covered_until,
            rates(&op.rates)
        );
    }
    out
}

pub fn dump() -> String {
    let mut out = String::new();
    for r in ROW_RATES {
        let _ = writeln!(out, "{}\tlist\t{}", r.model, card(&r.list));
        for cc in r.cost {
            let p = providers::by_id(cc.provider).name;
            match &cc.card {
                CostCard::List => {
                    let _ = writeln!(out, "{}\t{p}\tlist", r.model);
                }
                CostCard::Own(c) => {
                    let _ = writeln!(out, "{}\t{p}\t{}", r.model, card(c));
                }
                CostCard::Unverified(why) => {
                    let _ = writeln!(out, "{}\t{p}\tunverified: {why}", r.model);
                }
                CostCard::OpenRouter(eps) => {
                    for e in *eps {
                        let _ = writeln!(
                            out,
                            "{}\t{p}\t{} | {} | {}\t{}",
                            r.model,
                            e.host,
                            e.tag,
                            e.class.as_str(),
                            card(&e.card)
                        );
                    }
                }
            }
        }
    }
    out
}

/// The compiled table as the input of the independent reference pricer
/// (`verify/pricing_reference.py`): every rate a decimal USD string per million tokens, every
/// multiplier a decimal, every day an ISO date, so the reference shares no arithmetic with this
/// crate, only the rates. `rates-sync dump --json` prints it.
pub fn spec_json() -> String {
    use serde_json::{Map, Value, json};
    fn day(d: u32) -> String {
        crate::fetch::date_of(i64::from(d) * 86_400)
    }
    fn tr(r: &TokenRates) -> Value {
        json!({
            "input": usd(r.input),
            "output": usd(r.output),
            "cache_read": usd(r.cache_read),
            "cache_write": usd(r.cache_write_5m),
            "cache_write_1h": r.cache_write_1h.map(usd),
        })
    }
    fn card_json(c: &Card) -> Value {
        let mut m = Map::new();
        m.insert("standard".into(), tr(&c.standard));
        for (k, v) in [
            ("long", c.long),
            ("fast", c.fast),
            ("fast_long", c.fast_long),
            ("ultrafast", c.ultrafast),
            ("ultrafast_long", c.ultrafast_long),
            ("flex", c.flex),
            ("flex_long", c.flex_long),
        ] {
            if let Some(r) = v {
                m.insert(k.into(), tr(&r));
            }
        }
        if let Some(lc) = c.long_context {
            m.insert(
                "long_context".into(),
                json!({"threshold": lc.threshold, "inclusive": lc.inclusive}),
            );
        }
        if let Some(g) = c.geo_us {
            m.insert("geo_us".into(), json!(bps_decimal(u64::from(g))));
        }
        if !c.tools.is_empty() {
            let t: Map<String, Value> = c
                .tools
                .iter()
                .map(|(k, f)| (k.as_str().to_owned(), json!(usd(*f))))
                .collect();
            m.insert("tools".into(), Value::Object(t));
        }
        if let Some(op) = c.off_peak {
            m.insert(
                "off_peak".into(),
                json!({
                    "peak": op.peak,
                    "weekdays_only": op.weekdays_only,
                    "holidays": op.holidays.iter().map(|&d| day(d)).collect::<Vec<_>>(),
                    "covered": [day(op.covered_from), day(op.covered_until)],
                    "rates": tr(&op.rates),
                }),
            );
        }
        Value::Object(m)
    }
    let mut cards = Map::new();
    let mut openrouter = Map::new();
    let mut pricing = Vec::new();
    for r in ROW_RATES {
        let list = format!("list:{}", r.model);
        cards.insert(list.clone(), card_json(&r.list));
        let mut cost = Map::new();
        for cc in r.cost {
            let p = providers::by_id(cc.provider).name;
            let v = match &cc.card {
                CostCard::List => "list".to_owned(),
                CostCard::Own(c) => {
                    let name = format!("{p}:{}", r.model);
                    cards.insert(name.clone(), card_json(c));
                    name
                }
                CostCard::Unverified(why) => format!("unverified: {why}"),
                CostCard::OpenRouter(eps) => {
                    let eps: Vec<Value> = eps
                        .iter()
                        .map(|e| {
                            let mut m = Map::new();
                            m.insert("host".into(), json!(e.host));
                            m.insert("tag".into(), json!(e.tag));
                            m.insert("cls".into(), json!(e.class.as_str()));
                            m.insert("standard".into(), tr(&e.card.standard));
                            if let (Some(l), Some(lc)) = (e.card.long, e.card.long_context) {
                                m.insert("long".into(), tr(&l));
                                m.insert(
                                    "long_context".into(),
                                    json!({"threshold": lc.threshold, "inclusive": lc.inclusive}),
                                );
                            }
                            if let Some((_, f)) = e
                                .card
                                .tools
                                .iter()
                                .find(|(k, _)| k.as_str() == "web_search")
                            {
                                m.insert("web_search".into(), json!(usd(*f)));
                            }
                            Value::Object(m)
                        })
                        .collect();
                    openrouter.insert(r.model.to_owned(), Value::Array(eps));
                    format!("openrouter:{}", r.model)
                }
            };
            cost.insert(p.to_owned(), json!(v));
        }
        pricing.push(json!({"model": r.model, "customer": list, "cost": cost}));
    }
    let doc = json!({
        "rate_version": providers::rates::RATE_VERSION,
        "providers": providers::PROVIDERS.iter().map(|p| p.name).collect::<Vec<_>>(),
        "openrouter_credit_fee": bps_decimal(u64::from(providers::rates::OPENROUTER_CREDIT_FEE)),
        "cards": cards,
        "openrouter": openrouter,
        "pricing": pricing,
    });
    serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n"
}

/// `11000` basis points → `1.1`.
fn bps_decimal(bps: u64) -> String {
    let whole = bps / 10_000;
    let mut frac = format!("{:04}", bps % 10_000);
    while frac.ends_with('0') {
        frac.pop();
    }
    if frac.is_empty() {
        whole.to_string()
    } else {
        format!("{whole}.{frac}")
    }
}
