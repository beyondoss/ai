//! The rate tables (`crates/providers/src/rates.rs`) against their primary sources
//! (`verify/catalog_truth.toml`: `[[card]]`, `[[pricing]]`, `[[openrouter]]`, `[openrouter_fee]`),
//! and against the catalog they price.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use providers::catalog::MODEL_ROUTES;
use providers::pricing::{Card, Class, TokenRates, Tool, usd};
use providers::rates::{self, CostCard, OrEndpoint, RATE_VERSION, ROW_RATES};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

fn truth() -> toml::Table {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../verify/catalog_truth.toml"
    );
    std::fs::read_to_string(path)
        .expect("read verify/catalog_truth.toml")
        .parse()
        .expect("verify/catalog_truth.toml parses")
}

fn array<'a>(t: &'a toml::Table, key: &str) -> &'a [toml::Value] {
    t.get(key)
        .and_then(toml::Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn s<'a>(t: &'a toml::Table, key: &str) -> &'a str {
    t.get(key)
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("{key} in {t:?}"))
}

/// Days since 1970-01-01 for a `YYYY-MM-DD`.
fn day(date: &str) -> u32 {
    let p: Vec<i64> = date.split('-').map(|x| x.parse().expect(date)).collect();
    let (y, m, d) = (p[0], p[1], p[2]);
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    u32::try_from(era * 146_097 + doe - 719_468).expect(date)
}

// --- one canonical text for a card, from either side ------------------------------------------

fn canon_rates(r: &TokenRates) -> String {
    format!(
        "{}/{}/{}/{}/{:?}",
        r.input, r.output, r.cache_read, r.cache_write_5m, r.cache_write_1h
    )
}

fn canon_card(c: &Card) -> String {
    let mut out = format!("standard={}", canon_rates(&c.standard));
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
            write!(out, " {k}={}", canon_rates(&r)).unwrap();
        }
    }
    if let Some(lc) = c.long_context {
        write!(out, " long_context={}/{}", lc.threshold, lc.inclusive).unwrap();
    }
    if let Some(g) = c.geo_us {
        write!(out, " geo_us={g}").unwrap();
    }
    let mut tools: Vec<String> = c
        .tools
        .iter()
        .map(|(t, f)| format!("{}={f}", t.as_str()))
        .collect();
    tools.sort();
    if !tools.is_empty() {
        write!(out, " tools={}", tools.join(",")).unwrap();
    }
    if let Some(op) = c.off_peak {
        write!(
            out,
            " off_peak={:?}/{}/{:?}/{}-{}/{}",
            op.peak,
            op.weekdays_only,
            op.holidays,
            op.covered_from,
            op.covered_until,
            canon_rates(&op.rates)
        )
        .unwrap();
    }
    out
}

fn toml_rates(v: &toml::Value) -> String {
    let t = v.as_table().expect("rates table");
    let r = TokenRates {
        input: usd(s(t, "input")),
        output: usd(s(t, "output")),
        cache_read: usd(s(t, "cache_read")),
        cache_write_5m: usd(s(t, "cache_write")),
        cache_write_1h: t
            .get("cache_write_1h")
            .and_then(toml::Value::as_str)
            .map(usd),
    };
    canon_rates(&r)
}

/// USD per call as micro-dollars.
fn fee_micros(usd_per_call: &str) -> u64 {
    // A per-call fee is at most six decimals of a dollar.
    usd(usd_per_call)
}

fn toml_card(t: &toml::Table) -> String {
    let mut out = format!("standard={}", toml_rates(&t["standard"]));
    for k in [
        "long",
        "fast",
        "fast_long",
        "ultrafast",
        "ultrafast_long",
        "flex",
        "flex_long",
    ] {
        if let Some(v) = t.get(k) {
            write!(out, " {k}={}", toml_rates(v)).unwrap();
        }
    }
    if let Some(lc) = t.get("long_context").and_then(toml::Value::as_table) {
        write!(
            out,
            " long_context={}/{}",
            lc["threshold"].as_integer().unwrap(),
            lc["inclusive"].as_bool().unwrap()
        )
        .unwrap();
    }
    if let Some(g) = t.get("geo_us").and_then(toml::Value::as_str) {
        write!(out, " geo_us={}", usd(g) / 100).unwrap();
    }
    if let Some(tools) = t.get("tools").and_then(toml::Value::as_table) {
        let mut v: Vec<String> = tools
            .iter()
            .map(|(k, f)| {
                assert!(Tool::parse(k).is_some(), "unknown tool {k}");
                format!("{k}={}", fee_micros(f.as_str().unwrap()))
            })
            .collect();
        v.sort();
        write!(out, " tools={}", v.join(",")).unwrap();
    }
    if let Some(op) = t.get("off_peak").and_then(toml::Value::as_table) {
        let peak: Vec<(u16, u16)> = array(op, "peak_utc")
            .iter()
            .map(|w| {
                let w = w.as_str().unwrap();
                let (a, b) = w.split_once('-').unwrap();
                let min = |x: &str| {
                    let (h, m) = x.split_once(':').unwrap();
                    h.parse::<u16>().unwrap() * 60 + m.parse::<u16>().unwrap()
                };
                (min(a), min(b))
            })
            .collect();
        let holidays: Vec<u32> = array(op, "holidays")
            .iter()
            .map(|h| day(h.as_str().unwrap()))
            .collect();
        let covered = array(op, "covered");
        write!(
            out,
            " off_peak={:?}/{}/{:?}/{}-{}/{}",
            peak,
            op["weekdays_only"].as_bool().unwrap(),
            holidays,
            day(covered[0].as_str().unwrap()),
            day(covered[1].as_str().unwrap()),
            toml_rates(&op["rates"])
        )
        .unwrap();
    }
    out
}

fn canon_endpoint(e: &OrEndpoint) -> String {
    format!(
        "{}|{}|{}|{}",
        e.host,
        e.tag,
        e.class.as_str(),
        canon_card(&e.card)
    )
}

fn toml_endpoint(t: &toml::Table) -> String {
    let mut card = toml::Table::new();
    card.insert("standard".into(), t["standard"].clone());
    if let Some(l) = t.get("long") {
        card.insert("long".into(), l.clone());
        card.insert("long_context".into(), t["long_context"].clone());
    }
    if let Some(w) = t.get("web_search") {
        let mut tools = toml::Table::new();
        tools.insert("web_search".into(), w.clone());
        card.insert("tools".into(), toml::Value::Table(tools));
    }
    format!(
        "{}|{}|{}|{}",
        s(t, "host"),
        s(t, "tag"),
        s(t, "class"),
        toml_card(&card)
    )
}

fn assert_sourced(t: &toml::Table, what: &str) {
    let sources: Vec<&str> = match t.get("source") {
        Some(toml::Value::String(one)) => vec![one.as_str()],
        Some(toml::Value::Array(many)) => many.iter().filter_map(toml::Value::as_str).collect(),
        _ => vec![],
    };
    assert!(
        !sources.is_empty() && sources.iter().all(|u| u.starts_with("https://")),
        "{what}: needs https source URLs"
    );
    let date = t.get("date").and_then(toml::Value::as_str).unwrap_or("");
    assert!(date.len() == 10, "{what}: needs a date");
}

/// Every rate in `rates.rs` is the one recorded, with its source, in the truth file; every
/// recorded rate is used.
#[test]
fn rates_match_truth() {
    let t = truth();
    let cards: BTreeMap<&str, &toml::Table> = array(&t, "card")
        .iter()
        .map(|c| {
            let c = c.as_table().unwrap();
            let name = s(c, "name");
            assert_sourced(c, name);
            assert!(
                !c.get("source")
                    .and_then(toml::Value::as_array)
                    .is_some_and(|v| v
                        .iter()
                        .any(|u| u.as_str().unwrap_or("").contains("openrouter.ai"))),
                "{name}: a vendor card never comes from OpenRouter"
            );
            (name, c)
        })
        .collect();
    let or: BTreeMap<&str, &toml::Table> = array(&t, "openrouter")
        .iter()
        .map(|o| {
            let o = o.as_table().unwrap();
            let slug = s(o, "slug");
            assert_sourced(o, slug);
            (slug, o)
        })
        .collect();
    let pricing = array(&t, "pricing");
    assert_eq!(
        pricing.len(),
        ROW_RATES.len(),
        "one [[pricing]] per catalog row"
    );
    let (mut used_cards, mut used_or) = (BTreeSet::new(), BTreeSet::new());
    for (p, r) in pricing.iter().zip(ROW_RATES) {
        let p = p.as_table().unwrap();
        let model = s(p, "model");
        assert_eq!(model, r.model, "[[pricing]] in catalog order");
        let cust = s(p, "list_card");
        if cust == "list" {
            // The catalog's ListPrice alone: `list_card_standard_is_the_list_price` holds the
            // rates, and nothing else may be on the card.
            assert_eq!(
                canon_card(&Card::new(r.list.standard)),
                canon_card(&r.list),
                "{model}: a `list` card carries only the standard tier"
            );
        } else {
            let c = cards
                .get(cust)
                .unwrap_or_else(|| panic!("{model}: no [[card]] {cust}"));
            used_cards.insert(cust);
            assert_eq!(
                canon_card(&r.list),
                toml_card(c),
                "{model}: list card {cust}"
            );
        }
        let cost = p["cost"].as_table().unwrap();
        assert_eq!(
            cost.len(),
            r.cost.len(),
            "{model}: one cost entry per candidate provider"
        );
        for cc in r.cost {
            let name = providers::by_id(cc.provider).name;
            let want = cost
                .get(name)
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| panic!("{model}: no cost for {name}"));
            match &cc.card {
                CostCard::List => assert_eq!(want, "list", "{model} {name}"),
                CostCard::Own(card) => {
                    let c = cards
                        .get(want)
                        .unwrap_or_else(|| panic!("{model} {name}: no [[card]] {want}"));
                    used_cards.insert(want);
                    assert_eq!(canon_card(card), toml_card(c), "{model} {name}: {want}");
                }
                CostCard::OpenRouter(eps) => {
                    let slug = want
                        .strip_prefix("openrouter:")
                        .unwrap_or_else(|| panic!("{model} {name}: {want}"));
                    let o = or
                        .get(slug)
                        .unwrap_or_else(|| panic!("{model}: no [[openrouter]] {slug}"));
                    used_or.insert(slug);
                    let theirs: Vec<String> = array(o, "endpoints")
                        .iter()
                        .map(|e| toml_endpoint(e.as_table().unwrap()))
                        .collect();
                    let ours: Vec<String> = eps.iter().map(canon_endpoint).collect();
                    assert_eq!(ours, theirs, "{model}: OpenRouter {slug}");
                }
                CostCard::Unverified(reason) => {
                    assert_eq!(
                        want.strip_prefix("unverified: "),
                        Some(*reason),
                        "{model} {name}"
                    );
                }
            }
        }
    }
    let unused: Vec<_> = cards.keys().filter(|k| !used_cards.contains(*k)).collect();
    assert!(
        unused.is_empty(),
        "[[card]] entries no row uses: {unused:?}"
    );
    let unused: Vec<_> = or.keys().filter(|k| !used_or.contains(*k)).collect();
    assert!(
        unused.is_empty(),
        "[[openrouter]] entries no row uses: {unused:?}"
    );

    let fee = t["openrouter_fee"].as_table().unwrap();
    assert_sourced(fee, "openrouter_fee");
    assert_eq!(
        u64::from(rates::OPENROUTER_CREDIT_FEE),
        10_000 + usd(s(fee, "credit_fee")) / 100,
        "OPENROUTER_CREDIT_FEE"
    );
}

/// `ROW_RATES` is the catalog, row for row, and each row prices every provider that can serve it.
#[test]
fn every_catalog_row_and_candidate_is_priced() {
    assert_eq!(ROW_RATES.len(), MODEL_ROUTES.len());
    for (rr, route) in ROW_RATES.iter().zip(MODEL_ROUTES) {
        assert_eq!(rr.model, route.model, "ROW_RATES in MODEL_ROUTES order");
        let want: BTreeSet<_> = route
            .candidates
            .iter()
            .chain(route.responses)
            .map(|c| providers::by_id(c.provider).name)
            .collect();
        let got: BTreeSet<_> = rr
            .cost
            .iter()
            .map(|c| providers::by_id(c.provider).name)
            .collect();
        assert_eq!(
            got, want,
            "{}: a cost card per candidate provider",
            rr.model
        );
        assert!(rates::for_row(route.model).is_some());
    }
}

/// The list card's standard tier is the catalog's published list price (`GET /v1/models`).
#[test]
fn list_card_standard_is_the_list_price() {
    for (rr, route) in ROW_RATES.iter().zip(MODEL_ROUTES) {
        let p = route.price;
        let s = rr.list.standard;
        assert_eq!(
            (s.input, s.output, s.cache_read, s.cache_write_5m),
            (
                usd(p.input),
                usd(p.output),
                usd(p.cache_read),
                usd(p.cache_write)
            ),
            "{}",
            rr.model
        );
    }
}

/// An OpenRouter endpoint's class is the tier its tag names, and every card is internally sane:
/// no long tier without a threshold, no zero input rate on a generation model.
#[test]
fn cards_are_well_formed() {
    let check = |what: &str, c: &Card| {
        assert_eq!(
            c.long.is_some(),
            c.long_context.is_some(),
            "{what}: long rates need a threshold"
        );
        assert!(c.standard.input > 0, "{what}: zero input rate");
    };
    for rr in ROW_RATES {
        check(rr.model, &rr.list);
        for cc in rr.cost {
            match &cc.card {
                CostCard::Own(c) => check(rr.model, c),
                CostCard::OpenRouter(eps) => {
                    assert!(!eps.is_empty(), "{}: no endpoints", rr.model);
                    for e in *eps {
                        check(e.tag, &e.card);
                        let last = e.tag.rsplit('/').next().unwrap_or("");
                        let want = match last {
                            "fast" | "priority" => Class::Fast,
                            "flex" => Class::Flex,
                            "ultrafast" => Class::Ultrafast,
                            _ => Class::Standard,
                        };
                        assert_eq!(e.class, want, "{}: {}", rr.model, e.tag);
                    }
                }
                CostCard::List | CostCard::Unverified(_) => {}
            }
        }
    }
}

/// FNV-1a 64 over the table's `Debug` text and the OpenRouter fee.
fn table_hash() -> String {
    let text = format!("{ROW_RATES:?}{}", rates::OPENROUTER_CREDIT_FEE);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// `RATE_VERSION` names exactly this table: any rate change must bump it, so a logged row always
/// identifies the rates that priced it.
#[test]
fn rate_version_names_this_table() {
    let hash = table_hash();
    let (date, recorded) = RATE_VERSION
        .split_once('.')
        .expect("RATE_VERSION is {date}.{hash}");
    assert_eq!(date.len(), 10, "RATE_VERSION date");
    assert_eq!(
        recorded, hash,
        "the rate table changed: set RATE_VERSION to \"{{check date}}.{hash}\""
    );
}
