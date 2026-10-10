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
    assert!(vendors::bedrock(&b, "Claude Haiku 4.5").is_ok());
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
