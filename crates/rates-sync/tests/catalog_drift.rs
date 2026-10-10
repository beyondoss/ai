//! `rates-sync catalog-drift` on fixtures: a fictional Claude line (Sonata 4.7, then 4.8) priced
//! from the committed Anthropic page with two rows added, vendor listings written here, and the
//! committed truth file with the fixture rows' entries appended. One case per rule: no change, a
//! successor (and its row, byte for byte), each needs-human reason, and each retirement.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use providers::catalog::{
    Candidate, IN_FILE, IN_IMAGE, IN_TEXT, ListPrice, ModelCard, ModelRoute, REASONING,
    STRUCTURED_OUTPUTS, TOOLS,
};
use providers::{ProviderId, WireFormat};
use rates_sync::catalog_drift::{Catalog, Class, Finding, Inputs, detect};
use rates_sync::catalog_edit::{self, Edit};
use rates_sync::snapshot::{self, Store};
use rates_sync::{SOURCES, TRUTH, read_spec, repo_root, sources};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const TODAY: &str = "2026-10-10";
const PRED_CREATED: u64 = 1_777_593_600; // 2026-05-01
const NEW_CREATED: u64 = 1_791_331_200; // 2026-10-07

const fn cands(native: &'static str, or: &'static str) -> [Candidate; 2] {
    [
        Candidate {
            provider: ProviderId::Anthropic,
            upstream_model: native,
            path: "/v1/messages",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: or,
            path: "/api/v1/chat/completions",
        },
    ]
}

const C45: [Candidate; 2] = cands("claude-sonata-4-5", "anthropic/claude-sonata-4.5");
const C46: [Candidate; 2] = cands("claude-sonata-4-6", "anthropic/claude-sonata-4.6");
const C47: [Candidate; 2] = cands("claude-sonata-4-7", "anthropic/claude-sonata-4.7");
const C48: [Candidate; 2] = cands("claude-sonata-4-8", "anthropic/claude-sonata-4.8");

fn row(model: &'static str, c: &'static [Candidate], name: &'static str) -> ModelRoute {
    ModelRoute {
        model,
        wire: WireFormat::Anthropic,
        candidates: c,
        responses: &[],
        price: ListPrice {
            input: "2",
            output: "10",
            cache_read: "0.2",
            cache_write: "2.5",
        },
        card: ModelCard {
            name,
            owned_by: "anthropic",
            created: PRED_CREATED,
            context_window: 200_000,
            max_output_tokens: 64_000,
            max_output_published: true,
            input: IN_TEXT | IN_IMAGE | IN_FILE,
            features: TOOLS | REASONING | STRUCTURED_OUTPUTS,
        },
    }
}

fn sonata47() -> ModelRoute {
    row("claude-sonata-4-7", &C47, "Claude Sonata 4.7")
}

fn sonata48() -> ModelRoute {
    row("claude-sonata-4-8", &C48, "Claude Sonata 4.8")
}

const FIXTURE_TRUTH: &str = r#"
[[row]]
model = "claude-sonata-4-7"
source = ["https://platform.claude.com/docs/en/about-claude/pricing.md"]
date = "2026-10-01"
input = "2"
output = "10"
features_present = ["tools", "reasoning"]

[[card]]
name = "anthropic:claude-sonata-4-7"
row = "Claude Sonata 4.7"

[[pricing]]
model = "claude-sonata-4-7"
list_card = "anthropic:claude-sonata-4-7"
cost = { anthropic = "list", openrouter = "openrouter:anthropic/claude-sonata-4.7" }

[[card]]
name = "anthropic:claude-sonata-4-8"
row = "Claude Sonata 4.8"

[[pricing]]
model = "claude-sonata-4-8"
list_card = "anthropic:claude-sonata-4-8"
cost = { anthropic = "list", openrouter = "openrouter:anthropic/claude-sonata-4.8" }
"#;

/// The committed truth file with the fixture rows' entries.
fn truth_text() -> String {
    std::fs::read_to_string(repo_root().join(TRUTH)).unwrap() + FIXTURE_TRUTH
}

fn anthropic_row(name: &str) -> String {
    format!("| {name} | $2 / MTok | $2.50 / MTok | $4 / MTok | $0.20 / MTok | $10 / MTok |")
}

/// One case's world: what each host lists, and which Anthropic rows the pricing page has.
struct World {
    rows: Vec<ModelRoute>,
    /// OpenRouter models: (slug, created, context, expiration).
    or: Vec<(&'static str, u64, u64, Option<&'static str>)>,
    /// Anthropic `/v1/models`: (id, created_at, retires_at).
    anthropic: Vec<(&'static str, &'static str, Option<&'static str>)>,
    /// Anthropic pricing rows added to the committed page.
    priced: Vec<&'static str>,
    /// Extra Together models.
    together: Vec<&'static str>,
    deprecations: Option<&'static str>,
}

fn base() -> World {
    World {
        rows: vec![sonata47()],
        or: vec![("anthropic/claude-sonata-4.7", PRED_CREATED, 200_000, None)],
        anthropic: vec![("claude-sonata-4-7", "2026-05-01T00:00:00Z", None)],
        priced: vec!["Claude Sonata 4.7"],
        together: Vec::new(),
        deprecations: None,
    }
}

/// `base()` plus Sonata 4.8 everywhere: the successor case.
fn with_48() -> World {
    let mut w = base();
    w.or.push(("anthropic/claude-sonata-4.8", NEW_CREATED, 200_000, None));
    w.anthropic
        .push(("claude-sonata-4-8", "2026-10-07T00:00:00Z", None));
    w.priced.push("Claude Sonata 4.8");
    w
}

fn or_model(slug: &str, created: u64, context: u64, exp: Option<&str>) -> Value {
    json!({
        "id": slug,
        "name": slug,
        "created": created,
        "context_length": context,
        "architecture": { "input_modalities": ["text", "image", "file"] },
        "supported_parameters": ["max_tokens", "reasoning", "response_format", "structured_outputs", "tool_choice", "tools"],
        "top_provider": { "max_completion_tokens": 64000 },
        "pricing": { "prompt": "0.000002", "completion": "0.00001" },
        "expiration_date": exp,
    })
}

fn endpoints(slug: &str) -> String {
    json!({ "data": { "id": slug, "endpoints": [{
        "provider_name": "Anthropic",
        "tag": "anthropic",
        "pricing": {
            "prompt": "0.000002",
            "completion": "0.00001",
            "input_cache_read": "0.0000002",
            "input_cache_write": "0.0000025"
        },
        "quantization": null,
        "context_length": 200000,
        "max_prompt_tokens": null
    }]}})
    .to_string()
}

fn run(w: &World) -> Vec<Finding> {
    let spec = rates_sync::spec::parse(&truth_text()).unwrap();
    let truth: toml::Table = truth_text().parse().unwrap();
    let committed = read_spec(&repo_root()).unwrap();
    let store = snapshot::load(
        &repo_root().join(SOURCES),
        &sources::all(&committed.openrouter_slugs()),
    )
    .unwrap();
    let mut text = store.text.clone();
    // The fixture rows on Anthropic's pricing page, after Haiku 4.5.
    let page = text.get_mut("anthropic.pricing").unwrap();
    let at = page.find("| Claude Haiku 4.5 ").unwrap();
    let eol = at + page[at..].find('\n').unwrap() + 1;
    let added: String = w.priced.iter().map(|n| anthropic_row(n) + "\n").collect();
    page.insert_str(eol, &added);
    if !w.together.is_empty() {
        let t = text.get_mut("together.models").unwrap();
        let mut v: Value = serde_json::from_str(t).unwrap();
        for id in &w.together {
            v.as_array_mut()
                .unwrap()
                .push(json!({ "id": id, "type": "chat" }));
        }
        *t = v.to_string();
    }
    let inputs = Inputs {
        today: TODAY.into(),
        store: Store {
            dir: store.dir.clone(),
            entries: store.entries.clone(),
            text,
        },
        openrouter: json!({ "data": w.or.iter().map(|(s, c, x, e)| or_model(s, *c, *x, *e)).collect::<Vec<_>>() })
            .to_string(),
        endpoints: w
            .or
            .iter()
            .map(|(s, ..)| ((*s).to_owned(), endpoints(s)))
            .collect(),
        anthropic: Some(
            json!({ "data": w.anthropic.iter().map(|(id, c, r)| json!({
                "id": id, "created_at": c, "retires_at": r
            })).collect::<Vec<_>>() })
            .to_string(),
        ),
        openai: None,
        bedrock: None,
        deprecations: w
            .deprecations
            .map(|d| BTreeMap::from([("anthropic".to_owned(), d.to_owned())]))
            .unwrap_or_default(),
    };
    let cat = Catalog {
        rows: &w.rows,
        spec: &spec,
        truth: &truth,
    };
    let a = detect(&inputs, &cat).unwrap();
    // Deterministic: the same inputs give the same findings, byte for byte.
    let b = detect(&inputs, &cat).unwrap();
    assert_eq!(
        a.iter()
            .map(|f| f.to_json().to_string())
            .collect::<Vec<_>>(),
        b.iter()
            .map(|f| f.to_json().to_string())
            .collect::<Vec<_>>()
    );
    a
}

fn only(fs: &[Finding]) -> &Finding {
    match fs {
        [f] => f,
        _ => panic!(
            "want one finding, got {:#?}",
            fs.iter()
                .map(|f| (&f.slug, f.class, &f.reasons))
                .collect::<Vec<_>>()
        ),
    }
}

fn human(fs: &[Finding], slug: &str, reason: &str) {
    let f = fs
        .iter()
        .find(|f| f.slug == slug)
        .unwrap_or_else(|| panic!("no finding {slug}: {fs:#?}"));
    assert_eq!(f.class, Class::NeedsHuman, "{slug}: {:?}", f.reasons);
    assert!(
        f.edit.is_none(),
        "{slug}: a needs-human finding has no edit"
    );
    assert!(
        f.reasons.iter().any(|r| r.starts_with(reason)),
        "{slug}: no reason starting {reason:?} in {:?}",
        f.reasons
    );
}

#[test]
fn no_change_finds_nothing() {
    let fs = run(&base());
    assert!(fs.is_empty(), "{fs:#?}");
}

const SONATA_48_ROW: &str = r#"    ModelRoute {
        model: "claude-sonata-4-8",
        wire: WireFormat::Anthropic,
        candidates: &[
            Candidate {
                provider: ProviderId::Anthropic,
                upstream_model: "claude-sonata-4-8",
                path: "/v1/messages",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                upstream_model: "anthropic/claude-sonata-4.8",
                path: "/api/v1/chat/completions",
            },
        ],
        responses: &[],
        price: price("2", "10", "0.2", "2.5"),
        card: card(
            "Claude Sonata 4.8",
            "anthropic",
            1791331200,
            200_000,
            64_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
"#;

#[test]
fn a_newer_version_on_the_same_hosts_is_a_successor_with_a_generated_row() {
    let fs = run(&with_48());
    let f = only(&fs);
    assert_eq!(f.class, Class::Successor, "{:?}", f.reasons);
    assert_eq!(f.slug, "add-claude-sonata-4-8");
    assert_eq!(f.model, "claude-sonata-4-8");
    assert_eq!(f.providers, ["anthropic", "openrouter"]);
    let Some(Edit::Add(a)) = &f.edit else {
        panic!("no edit")
    };
    assert_eq!(a.row, SONATA_48_ROW);
    assert_eq!(a.after.as_deref(), Some("claude-sonata-4-7"));
    assert_eq!(
        a.pricing,
        "[[pricing]]\nmodel = \"claude-sonata-4-8\"\nlist_card = \"anthropic:claude-sonata-4-8\"\ncost = { anthropic = \"list\", openrouter = \"openrouter:anthropic/claude-sonata-4.8\" }\n"
    );
    assert_eq!(
        a.cards,
        [(
            "anthropic:claude-sonata-4-7".to_owned(),
            "[[card]]\nname = \"anthropic:claude-sonata-4-8\"\nrow = \"Claude Sonata 4.8\"\n"
                .to_owned()
        )]
    );
    assert_eq!(
        a.truth_row,
        Some((
            "claude-sonata-4-7".to_owned(),
            "[[row]]\nmodel = \"claude-sonata-4-8\"\nsource = [\"https://platform.claude.com/docs/en/about-claude/pricing.md\"]\ndate = \"2026-10-10\"\ninput = \"2\"\noutput = \"10\"\ncache_read = \"0.2\"\ncache_write = \"2.5\"\ncreated = 1791331200\nfeatures_present = [\"tools\", \"reasoning\"]\n"
                .to_owned()
        ))
    );
    assert_eq!(a.new_slugs, ["anthropic/claude-sonata-4.8"]);
    assert!(a.refresh.contains(&"anthropic.pricing".to_owned()));
}

#[test]
fn the_generated_row_splices_in_after_its_predecessor() {
    let fs = run(&with_48());
    let edit = only(&fs).edit.clone().unwrap();
    let catalog = format!(
        "pub const MODEL_ROUTES: &[ModelRoute] = &[\n{}{}];\n",
        "    ModelRoute {\n        model: \"claude-sonata-4-7\",\n        wire: WireFormat::Anthropic,\n    },\n",
        "    ModelRoute {\n        model: \"claude-sonata-5\",\n        wire: WireFormat::Anthropic,\n    },\n",
    );
    let out = catalog_edit::apply_catalog(&catalog, &edit).unwrap();
    let a = out.find("\"claude-sonata-4-7\"").unwrap();
    let b = out.find("\"claude-sonata-4-8\"").unwrap();
    let c = out.find("\"claude-sonata-5\"").unwrap();
    assert!(a < b && b < c, "{out}");
    assert!(out.contains(SONATA_48_ROW));
    let truth = FIXTURE_TRUTH.trim_start().to_owned();
    let t = catalog_edit::apply_truth(&truth, &edit).unwrap();
    let parsed: toml::Table = t.parse().unwrap();
    let pricing: Vec<&str> = parsed["pricing"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["model"].as_str().unwrap())
        .collect();
    assert_eq!(
        pricing,
        [
            "claude-sonata-4-7",
            "claude-sonata-4-8",
            "claude-sonata-4-8"
        ],
        "the new [[pricing]] follows its predecessor's"
    );
}

#[test]
fn needs_human_new_family() {
    let mut w = base();
    w.or.push(("anthropic/claude-cadenza-1", NEW_CREATED, 200_000, None));
    human(
        &run(&w),
        "review-openrouter-anthropic-claude-cadenza-1",
        "new family",
    );
}

#[test]
fn needs_human_ambiguous_lineage() {
    let mut w = base();
    // Released after 4.7, versioned below it.
    w.or.push(("anthropic/claude-sonata-4.6", NEW_CREATED, 200_000, None));
    human(
        &run(&w),
        "review-openrouter-anthropic-claude-sonata-4.6",
        "ambiguous lineage",
    );
}

#[test]
fn needs_human_new_host() {
    let mut w = with_48();
    w.together.push("anthropic/Claude-Sonata-4.8");
    human(&run(&w), "add-claude-sonata-4-8", "new host: together");
}

#[test]
fn needs_human_feature_class() {
    let mut w = with_48();
    w.or[1].2 = 1_000_000;
    human(
        &run(&w),
        "add-claude-sonata-4-8",
        "feature class differs: context window 200000 → 1000000",
    );
}

#[test]
fn needs_human_no_rate() {
    let mut w = with_48();
    w.priced.pop();
    human(
        &run(&w),
        "add-claude-sonata-4-8",
        "no rate: anthropic:claude-sonata-4-8",
    );
}

#[test]
fn needs_human_primary_not_listed() {
    let mut w = with_48();
    w.anthropic.pop();
    human(
        &run(&w),
        "add-claude-sonata-4-8",
        "the primary host anthropic does not list version 4.8",
    );
}

#[test]
fn a_failover_retiring_inside_the_window_leaves_the_row() {
    let mut w = base();
    w.or[0].3 = Some("2026-10-21");
    let fs = run(&w);
    let f = only(&fs);
    assert_eq!(f.class, Class::Retirement, "{:?}", f.reasons);
    assert_eq!(f.slug, "retire-claude-sonata-4-7");
    assert_eq!(f.providers, ["anthropic"]);
    let Some(Edit::Retire(r)) = &f.edit else {
        panic!("no edit")
    };
    assert_eq!(
        r.candidates.as_deref(),
        Some(
            "&[Candidate {\n            provider: ProviderId::Anthropic,\n            upstream_model: \"claude-sonata-4-7\",\n            path: \"/v1/messages\",\n        }]"
        )
    );
    assert_eq!(r.cost.as_deref(), Some("cost = { anthropic = \"list\" }"));
    assert!(r.alias_to.is_none());
    assert!(r.drop_cards.is_empty());
    assert_eq!(r.retired.len(), 1);
    assert!(
        r.retired[0].starts_with(
            "[[retired]]\nprovider = \"openrouter\"\nid = \"anthropic/claude-sonata-4.7\"\nsource = \"https://openrouter.ai/api/v1/models?output_modalities=all\"\nretires = \"2026-10-21\"\nnote = "
        ),
        "{}",
        r.retired[0]
    );
    // The truth edit keeps the file parseable, with the cost and the retirement recorded.
    let t = catalog_edit::apply_truth(&truth_text(), f.edit.as_ref().unwrap()).unwrap();
    assert!(t.contains("model = \"claude-sonata-4-7\"\nlist_card = \"anthropic:claude-sonata-4-7\"\ncost = { anthropic = \"list\" }\n"));
    let _: toml::Table = t.parse().unwrap();
}

#[test]
fn a_retirement_past_the_window_waits() {
    let mut w = base();
    w.or[0].3 = Some("2026-11-30");
    assert!(run(&w).is_empty());
}

#[test]
fn needs_human_primary_retires() {
    let mut w = base();
    w.deprecations = Some(
        "## Model status\n\n| API model name | Current state | Deprecated | Tentative retirement date |\n| :- | :- | :- | :- |\n| claude-sonata-4-7-20260501 | Deprecated | August 1, 2026 | October 20, 2026 |\n",
    );
    human(
        &run(&w),
        "retire-claude-sonata-4-7",
        "the primary anthropic/claude-sonata-4-7 retires",
    );
}

#[test]
fn a_whole_row_retiring_becomes_an_alias_of_its_successor() {
    let mut w = with_48();
    w.rows.push(sonata48());
    w.anthropic[0].2 = Some("2026-10-20T00:00:00Z");
    w.or[0].3 = Some("2026-10-20");
    let fs = run(&w);
    let f = only(&fs);
    assert_eq!(f.class, Class::Retirement, "{:?}", f.reasons);
    let Some(Edit::Retire(r)) = &f.edit else {
        panic!("no edit")
    };
    assert_eq!(r.alias_to.as_deref(), Some("claude-sonata-4-8"));
    assert!(r.candidates.is_none() && r.cost.is_none());
    assert_eq!(r.drop_cards, ["anthropic:claude-sonata-4-7"]);
    assert!(
        r.retired[0]
            .starts_with("[[retired]]\nmodel = \"claude-sonata-4-7\"\nprovider = \"anthropic\"\n")
    );
    assert_eq!(f.providers, ["anthropic", "openrouter"]);
    let t = catalog_edit::apply_truth(&truth_text(), f.edit.as_ref().unwrap()).unwrap();
    assert!(!t.contains("model = \"claude-sonata-4-7\"\nlist_card"));
    let _: toml::Table = t.parse().unwrap();
}

#[test]
fn needs_human_whole_row_without_successor() {
    let mut w = base();
    w.anthropic[0].2 = Some("2026-10-20T00:00:00Z");
    w.or[0].3 = Some("2026-10-20");
    human(
        &run(&w),
        "retire-claude-sonata-4-7",
        "every candidate of claude-sonata-4-7 retires",
    );
}

#[test]
fn needs_human_when_a_listing_loses_many_candidates_at_once() {
    let mut w = base();
    w.rows = vec![
        row("claude-sonata-4-5", &C45, "Claude Sonata 4.5"),
        row("claude-sonata-4-6", &C46, "Claude Sonata 4.6"),
        sonata47(),
    ];
    w.or.clear();
    w.anthropic = vec![
        ("claude-sonata-4-5", "2026-01-01T00:00:00Z", None),
        ("claude-sonata-4-6", "2026-03-01T00:00:00Z", None),
        ("claude-sonata-4-7", "2026-05-01T00:00:00Z", None),
    ];
    let fs = run(&w);
    human(
        &fs,
        "vanished-openrouter",
        "3 candidates vanished from openrouter's listing at once",
    );
    assert!(fs.iter().all(|f| f.class == Class::NeedsHuman), "{fs:#?}");
}

#[test]
fn a_candidate_its_host_stopped_listing_retires_today() {
    let mut w = base();
    w.or.clear();
    let f = run(&w);
    let f = only(&f);
    assert_eq!(f.class, Class::Retirement, "{:?}", f.reasons);
    let Some(Edit::Retire(r)) = &f.edit else {
        panic!("no edit")
    };
    assert!(
        r.retired[0].contains("retired = \"2026-10-10\""),
        "{}",
        r.retired[0]
    );
}
