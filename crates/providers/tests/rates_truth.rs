//! The rate table (`crates/providers/src/rates/generated.rs`) against the catalog it prices, and
//! its version. That the table is exactly what the primary sources generate is
//! `rates_generated_from_sources` (crates/rates-sync/tests/regenerate.rs).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use providers::catalog::MODEL_ROUTES;
use providers::pricing::{Card, Class, usd};
use providers::rates::{self, CostCard, RATE_VERSION, ROW_RATES};
use std::collections::BTreeSet;

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
