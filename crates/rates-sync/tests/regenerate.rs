//! The committed rate table is exactly what the committed snapshots generate, offline and
//! byte for byte; and each source's reader fails loudly, never guesses, on a layout it does not
//! expect.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rates_sync::{GENERATED, SOURCES, diff, repo_root, vendors};

fn snapshot(file: &str) -> String {
    std::fs::read_to_string(repo_root().join(SOURCES).join(file)).unwrap()
}

/// Regenerate `crates/providers/src/rates/generated.rs` from `verify/rates_sources/` and
/// `verify/catalog_truth.toml`, and diff it with the committed file. Every snapshot's sha256 is
/// checked against the manifest on the way.
#[test]
fn rates_generated_from_sources() {
    let root = repo_root();
    let want = rates_sync::generate(&root, &root.join(SOURCES)).unwrap_or_else(|e| panic!("{e}"));
    let have = std::fs::read_to_string(root.join(GENERATED)).unwrap();
    if let Some(d) = diff::lines(&have, &want) {
        panic!(
            "{GENERATED} is not what the snapshots generate; run `mise run rates:generate`:\n{d}"
        );
    }
    // A committed [[conflict]] whose sources agree again is stale: remove it.
    let spec = rates_sync::read_spec(&root).unwrap();
    let store = rates_sync::snapshot::load(
        &root.join(SOURCES),
        &rates_sync::sources::all(&spec.openrouter_slugs()),
    )
    .unwrap();
    let table = rates_sync::build::build(&store, &spec).unwrap();
    assert!(
        table.settled_conflicts.is_empty(),
        "[[conflict]] entries whose sources agree again, to remove from verify/catalog_truth.toml: {:?}",
        table.settled_conflicts
    );
    // Twice is the same: nothing in generation depends on time, order of a map, or the network.
    assert_eq!(
        want,
        rates_sync::generate(&root, &root.join(SOURCES)).unwrap()
    );
}

/// A changed header, a missing row, or a malformed cell is an error.
#[test]
fn readers_fail_loudly_on_layout_changes() {
    let a = snapshot("anthropic/pricing.md");
    assert!(
        vendors::anthropic(&a)
            .unwrap()
            .model("Claude Opus 4.8")
            .is_ok()
    );
    let renamed = a.replace("| Cache hits and refreshes |", "| Cache reads |");
    assert!(vendors::anthropic(&renamed).is_err(), "a renamed column");
    assert!(
        vendors::anthropic(&a)
            .unwrap()
            .model("Claude Opus 9")
            .is_err(),
        "a missing row"
    );
    let bad = a.replace("$6.25 / MTok", "$6.25 per MTok");
    assert!(
        vendors::anthropic(&bad)
            .unwrap()
            .model("Claude Opus 4.8")
            .is_err(),
        "a malformed cell"
    );

    let o = snapshot("openai/pricing.md");
    assert!(vendors::openai(&o).is_ok());
    assert!(
        vendors::openai(&o.replace("### Flex pricing data", "### Flexible pricing data")).is_err()
    );
    let dup = o.replace("| gpt-5-mini | $0.25 |", "| gpt-5-nano | $0.25 |");
    assert!(
        vendors::openai(&dup)
            .unwrap()
            .tier("standard", "gpt-5-nano")
            .is_err(),
        "a duplicate row"
    );

    let d = snapshot("deepseek/pricing.html");
    assert!(vendors::deepseek(&d).is_ok());
    assert!(vendors::deepseek(&d.replacen("<td>PEAK</td>", "<td>PEAK HOURS</td>", 1)).is_err());

    let x = snapshot("xai/pricing.md");
    assert!(vendors::xai_page(&x, "grok-4.7").is_ok());
    assert!(
        vendors::xai_page(
            &x.replace("(≥ 200k prompt tokens)", "(> 200k prompt tokens)"),
            "grok-4.7"
        )
        .is_err()
    );

    let g = snapshot("groq/models.md");
    assert!(vendors::groq(&g, "openai/gpt-oss-120b").is_ok());
    assert!(
        vendors::groq(
            &g.replace("$0.15 input$0.60 output", "$0.15 in / $0.60 out"),
            "openai/gpt-oss-120b"
        )
        .is_err()
    );

    let f = snapshot("fireworks/pricing.md");
    assert!(vendors::fireworks(&f, "Kimi K3", "kimi-k3").is_ok());
    assert!(
        vendors::fireworks(
            &f.replace(
                "| Model | Standard | Priority |",
                "| Model | Standard | Fast |"
            ),
            "Kimi K3",
            "kimi-k3"
        )
        .is_err()
    );

    let t = snapshot("together/models.json");
    assert!(vendors::together_api(&t, "zai-org/GLM-5.3").is_ok());
    assert!(vendors::together_api(&t, "zai-org/GLM-9").is_err());

    let b = snapshot("bedrock/foundation-models-us-east-1.json");
    let flat = vendors::bedrock(&b, "Claude Haiku 4.5").unwrap();
    assert_eq!(flat.regional_long, vendors::BedrockScope::default());
    let odd = b.replace(
        "\"USE1-MP:USE1_OutputTokenCount-Units\"",
        "\"USE1-MP:USE1_Mystery-Units\"",
    );
    assert_ne!(odd, b);
    assert!(
        vendors::bedrock(&odd, "Claude Haiku 4.5").is_err(),
        "an unrecognized usage type"
    );
}

/// Anthropic's fast mode table as served on 2026-10-18: the cells name the currency
/// (`$10 USD / MTok`), where the model table's cells stay `$10 / MTok`.
const ANTHROPIC_FAST_USD: &str = "\
| Model                           | Input          | Output         |
| ------------------------------- | -------------- | -------------- |
| Claude Opus 5.5                 | $8 USD / MTok  | $40 USD / MTok |
| Claude Opus 5 / Claude Opus 4.8 | $10 USD / MTok | $50 USD / MTok |";

/// `$x USD / MTok` reads as `$x / MTok`; any other currency or spacing is still an error.
#[test]
fn anthropic_fast_mode_cells_may_name_usd() {
    let a = snapshot("anthropic/pricing.md");
    let old = "\
| Model                           | Input      | Output     |
| ------------------------------- | ---------- | ---------- |
| Claude Opus 5.5                 | $8 / MTok  | $40 / MTok |
| Claude Opus 5 / Claude Opus 4.8 | $10 / MTok | $50 / MTok |";
    assert_eq!(a.matches(old).count(), 1);
    let usd = a.replace(old, ANTHROPIC_FAST_USD);
    let fast = |md: &str, name: &str| vendors::anthropic(md).unwrap().fast(name);
    for name in ["Claude Opus 5.5", "Claude Opus 5", "Claude Opus 4.8"] {
        let want = fast(&a, name).unwrap();
        assert!(want.is_some(), "{name}");
        assert_eq!(fast(&usd, name).unwrap(), want, "{name}");
    }
    for bad in [
        "$10 EUR / MTok",
        "$10 USD/MTok",
        "$10 USD USD / MTok",
        "10 USD / MTok",
    ] {
        let md = usd.replace("$10 USD / MTok", bad);
        assert!(fast(&md, "Claude Opus 4.8").is_err(), "{bad:?}");
    }
}

/// Claude Haiku 5.5 is priced by prompt length. Anthropic's model table splits it into two rows,
/// "(for prompts up to 100,000 tokens)" and "(for prompts over 100,000 tokens)", and AWS gives it
/// a second `long_ctx` SKU per token kind in each scope. Both readers keep the tiers apart, and
/// fail on a split they cannot read as one threshold.
#[test]
fn prompt_length_tiers_are_read_apart() {
    let a = snapshot("anthropic/pricing.md");
    let m = vendors::anthropic(&a)
        .unwrap()
        .model("Claude Haiku 5.5")
        .unwrap();
    let r = |s: &str| rates_sync::dec::Rate::per_million(s).unwrap();
    assert_eq!(m.standard.input, r("0.10"));
    assert_eq!(m.standard.cache_write_1h, Some(r("0.20")));
    let (n, long) = m.long.expect("a long tier");
    assert_eq!(n, 100_000);
    assert_eq!(
        (
            long.input,
            long.output,
            long.cache_read,
            long.cache_write,
            long.cache_write_1h
        ),
        (
            r("0.50"),
            Some(r("2.50")),
            Some(r("0.05")),
            Some(r("0.625")),
            Some(r("1"))
        )
    );
    // A flat model has no tier.
    let flat = vendors::anthropic(&a)
        .unwrap()
        .model("Claude Haiku 4.5")
        .unwrap();
    assert!(flat.long.is_none());
    // The two halves must name one threshold.
    let skew = a.replace(
        "Claude Haiku 5.5 (for prompts over 100,000 tokens)",
        "Claude Haiku 5.5 (for prompts over 200,000 tokens)",
    );
    assert_ne!(skew, a);
    assert!(
        vendors::anthropic(&skew)
            .unwrap()
            .model("Claude Haiku 5.5")
            .is_err()
    );
    // Half a split is not a model.
    let half = a.replace(
        "Claude Haiku 5.5 (for prompts over 100,000 tokens)",
        "Claude Haiku 5.6 (for prompts over 100,000 tokens)",
    );
    assert!(
        vendors::anthropic(&half)
            .unwrap()
            .model("Claude Haiku 5.5")
            .is_err()
    );
    // A label the reader does not know is an error, not a guess.
    let odd = a.replace(
        "(for prompts over 100,000 tokens)",
        "(for prompts beyond 100,000 tokens)",
    );
    assert!(
        vendors::anthropic(&odd)
            .unwrap()
            .model("Claude Haiku 5.5")
            .is_err()
    );

    let b = snapshot("bedrock/foundation-models-us-east-1.json");
    let h = vendors::bedrock(&b, "Claude Haiku 5.5").unwrap();
    assert_eq!(h.global.output, Some(r("0.5")));
    assert_eq!(h.global_long.output, Some(r("2.5")));
    assert_eq!(h.regional.output, Some(r("0.55")));
    assert_eq!(h.regional_long.output, Some(r("2.75")));
    assert_eq!(h.regional_long.cache_write_1h, Some(r("1.1")));
    // Two long-context output SKUs in one scope is still an error.
    // (The Global long output SKU, re-spelled as a second Regional one.)
    let dup = b.replace(
        "USE1_output_tokens_long_ctx_global_standard-Units",
        "USE1_output_tokens_long_ctx_standard_2-Units",
    );
    assert_ne!(dup, b);
    assert!(vendors::bedrock(&dup, "Claude Haiku 5.5").is_err());
}
