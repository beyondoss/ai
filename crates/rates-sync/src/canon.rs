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
