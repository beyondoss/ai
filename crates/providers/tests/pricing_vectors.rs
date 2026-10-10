//! The golden vectors (`verify/pricing_vectors.json`): every row must price exactly as recorded.
//! They were computed by an independent implementation in exact rationals, so a failure here is a
//! disagreement between two implementations of the contract, not a snapshot to refresh.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use providers::WireFormat;
use providers::pricing::{Priced, Tool, ToolCounts, Unpriced, UsageRow, price};
use providers::rates::RATE_VERSION;
use serde_json::Value;

fn vectors() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../verify/pricing_vectors.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("read pricing_vectors.json"))
        .expect("pricing_vectors.json parses")
}

/// A vector's row as the pricer reads it. Absent fields take the documented defaults. The row's
/// `server_tools` text is parsed as the gateway parses it, so a malformed one is an outcome.
fn row(v: &Value) -> Result<UsageRow<'_>, Unpriced> {
    let n = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    let s = |k: &str| v.get(k).and_then(Value::as_str);
    let b = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
    let tools = ToolCounts::parse(s("server_tools").unwrap_or(""))?;
    let known = [
        "price_model",
        "provider",
        "price_variant",
        "usage_wire",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_write_1h_tokens",
        "gateway_cache_write_tokens",
        "reasoning_tokens",
        "server_tools",
        "service_tier",
        "speed",
        "inference_geo",
        "upstream_cost_usd",
        "upstream_tool_cost_usd",
        "served_by",
        "usage_estimated",
        "upstream_may_continue",
        "cache_hit",
        "unix_secs",
    ];
    for k in v.as_object().expect("row is an object").keys() {
        assert!(known.contains(&k.as_str()), "unknown row field {k}");
    }
    Ok(UsageRow {
        price_model: s("price_model"),
        provider: s("provider"),
        price_variant: s("price_variant"),
        usage_wire: match s("usage_wire").unwrap_or("anthropic") {
            "openai" => WireFormat::OpenAi,
            "anthropic" => WireFormat::Anthropic,
            w => panic!("usage_wire {w}"),
        },
        input_tokens: n("input_tokens"),
        output_tokens: n("output_tokens"),
        cache_read_tokens: n("cache_read_tokens"),
        cache_write_tokens: n("cache_write_tokens"),
        cache_write_1h_tokens: n("cache_write_1h_tokens"),
        gateway_cache_write_tokens: n("gateway_cache_write_tokens"),
        server_tools: tools,
        service_tier: s("service_tier"),
        speed: s("speed"),
        inference_geo: s("inference_geo"),
        upstream_cost_usd: s("upstream_cost_usd"),
        upstream_tool_cost_usd: s("upstream_tool_cost_usd"),
        served_by: s("served_by"),
        usage_estimated: b("usage_estimated"),
        upstream_may_continue: b("upstream_may_continue"),
        cache_hit: b("cache_hit"),
        unix_secs: n("unix_secs"),
    })
}

fn got(p: &Priced) -> Value {
    serde_json::json!({
        "status": p.status.as_str(),
        "basis": p.basis.as_str(),
        "cost_micros": p.cost.micros,
        "price_micros": p.price.micros,
        "cost_class": p.cost.class.as_str(),
        "price_class": p.price.class.as_str(),
        "cost_long": p.cost.long,
        "price_long": p.price.long,
    })
}

#[test]
fn golden_vectors_price_exactly() {
    let doc = vectors();
    assert_eq!(
        doc["rate_version"], RATE_VERSION,
        "the vectors were computed against another rate table: recompute them"
    );
    let vs = doc["vectors"].as_array().expect("vectors");
    assert!(vs.len() >= 50, "the vectors cover every dimension");
    let mut failures = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    for v in vs {
        let name = v["name"].as_str().expect("name");
        assert!(names.insert(name), "{name}: duplicate vector name");
        assert!(
            v["why"].as_str().is_some_and(|w| !w.is_empty()),
            "{name}: says why"
        );
        let actual = match row(&v["row"]).and_then(|r| price(&r)) {
            Ok(p) => got(&p),
            Err(e) => serde_json::json!({ "unpriced": e.as_str() }),
        };
        if actual != v["expect"] {
            failures.push(format!("{name}: want {} got {actual}", v["expect"]));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every refusal code and every class appears in some vector, so the contract's whole surface is
/// pinned, not only the happy path.
#[test]
fn vectors_cover_every_outcome() {
    let doc = vectors();
    let vs = doc["vectors"].as_array().expect("vectors");
    let expects: Vec<&Value> = vs.iter().map(|v| &v["expect"]).collect();
    let has = |k: &str, val: &str| expects.iter().any(|e| e[k] == val);
    for code in [
        "no_price_model",
        "unknown_model",
        "unknown_provider",
        "not_a_candidate",
        "unknown_variant",
        "unknown_class",
        "unknown_geo",
        "no_rate",
        "no_tool_fee",
        "inconsistent_tokens",
        "bad_reported_cost",
        "unknown_host",
        "unknown_calendar",
        "overflow",
        "unknown_tool",
    ] {
        assert!(has("unpriced", code), "no vector refuses with {code}");
    }
    for class in ["standard", "fast", "ultrafast", "flex", "off_peak"] {
        assert!(has("cost_class", class), "no vector costs at {class}");
    }
    for basis in ["tokens", "reported", "dearest_endpoint", "cache_hit"] {
        assert!(has("basis", basis), "no vector with basis {basis}");
    }
    assert!(has("status", "estimated"));
    assert!(expects.iter().any(|e| e["cost_long"] == true));
    let tools: std::collections::BTreeSet<&str> = vs
        .iter()
        .filter_map(|v| v["row"]["server_tools"].as_str())
        .flat_map(|t| t.split(','))
        .filter_map(|kv| kv.split_once('=').map(|(k, _)| k))
        .collect();
    for t in Tool::ALL {
        assert!(tools.contains(t.as_str()), "no vector uses {}", t.as_str());
    }
}
