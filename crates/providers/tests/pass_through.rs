//! Pricing is pass-through (owner decision, 2026-10-10): the customer pays exactly what we pay
//! for each request, on the card of the host that served it, fees included. No row may be priced
//! below its cost, or above it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use providers::WireFormat;
use providers::pricing::{Tool, ToolCounts, UsageRow, price};
use providers::rates::ROW_RATES;
use serde_json::Value;

/// Every priced golden vector says price == cost, on the same class and tier.
#[test]
fn every_vector_prices_at_cost() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../verify/pricing_vectors.json"
    );
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut priced = 0;
    for v in doc["vectors"].as_array().unwrap() {
        let e = &v["expect"];
        if e.get("unpriced").is_some() {
            continue;
        }
        priced += 1;
        assert_eq!(e["price_micros"], e["cost_micros"], "{}", v["name"]);
        assert_eq!(e["price_class"], e["cost_class"], "{}", v["name"]);
        assert_eq!(e["price_long"], e["cost_long"], "{}", v["name"]);
    }
    assert!(priced >= 50);
}

/// Every row × every candidate provider × every class × a spread of usage shapes (short, long,
/// cached, 1-hour writes, gateway writes, tools, data residency, a reported cost, an estimate):
/// wherever the pricer prices, the price is the cost, part for part.
#[test]
fn every_card_prices_at_cost() {
    let mut checked = 0u32;
    for rr in ROW_RATES {
        for cc in rr.cost {
            let provider = providers::by_id(cc.provider).name;
            for wire in [WireFormat::Anthropic, WireFormat::OpenAi] {
                for (tier, speed) in [
                    (None, None),
                    (Some("priority"), None),
                    (None, Some("fast")),
                    (Some("flex"), None),
                    (Some("ultrafast"), None),
                ] {
                    for (input, read, write, w1h, gw) in [
                        (1_200u64, 0u64, 0u64, 0u64, 0u64),
                        (300_000, 0, 0, 0, 0),
                        (50_000, 40_000, 5_000, 2_000, 0),
                        (20_000, 0, 0, 0, 15_000),
                    ] {
                        for geo in [None, Some("us")] {
                            for reported in [None, Some("0.0123456")] {
                                let input_tokens = match wire {
                                    WireFormat::Anthropic => input,
                                    WireFormat::OpenAi => input + read + write,
                                };
                                let r = UsageRow {
                                    price_model: Some(rr.model),
                                    provider: Some(provider),
                                    usage_wire: wire,
                                    input_tokens,
                                    output_tokens: 777,
                                    cache_read_tokens: read,
                                    cache_write_tokens: write,
                                    cache_write_1h_tokens: w1h,
                                    gateway_cache_write_tokens: gw,
                                    server_tools: ToolCounts::new().with(Tool::WebSearch, 2),
                                    service_tier: tier,
                                    speed,
                                    inference_geo: geo,
                                    upstream_cost_usd: reported,
                                    unix_secs: 1_791_806_400, // a Monday, 12:00 UTC
                                    ..UsageRow::default()
                                };
                                if let Ok(p) = price(&r) {
                                    assert_eq!(
                                        p.price, p.cost,
                                        "{} on {provider}: {r:?}",
                                        rr.model
                                    );
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 1_000, "only {checked} priced combinations");
}
