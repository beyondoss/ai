//! The drift classifier (`rates-sync classify`) on snapshot pairs: the committed store, and the
//! same store with one edit a vendor could make. One case per rule.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rates_sync::classify::{Class, Report, classify};
use rates_sync::snapshot::{self, Entry, Store};
use rates_sync::{SOURCES, build, read_spec, repo_root, sources};
use std::collections::BTreeMap;

const KIMI: &str = r"| [Kimi K3](https://app.fireworks.ai/models/fireworks/kimi-k3) | \$3.00 / \$0.30 / \$15.00 | \$3.75 / \$0.375 / \$18.75 |";

fn committed() -> Store {
    let spec = read_spec(&repo_root()).unwrap();
    snapshot::load(
        &repo_root().join(SOURCES),
        &sources::all(&spec.openrouter_slugs()),
    )
    .unwrap()
}

/// Classify the committed store against a copy that `edit` changed.
fn run(edit: impl FnOnce(&mut BTreeMap<String, String>, &mut BTreeMap<String, Entry>)) -> Report {
    let spec = read_spec(&repo_root()).unwrap();
    let old = committed();
    let (mut text, mut entries) = (old.text.clone(), old.entries.clone());
    edit(&mut text, &mut entries);
    let new = Store {
        dir: old.dir.clone(),
        entries,
        text,
    };
    let a = build::build(&old, &spec).unwrap();
    let b = build::build(&new, &spec);
    classify(&a, b.as_ref().map_err(String::as_str), &new.entries)
}

fn replace(text: &mut BTreeMap<String, String>, id: &str, from: &str, to: &str) {
    let t = text.get_mut(id).unwrap();
    assert_eq!(t.matches(from).count(), 1, "{id}: {from:?}");
    *t = t.replacen(from, to, 1);
}

/// Edit the first OpenRouter endpoint list with more than one endpoint.
fn openrouter(text: &mut BTreeMap<String, String>, f: impl FnOnce(&mut Vec<serde_json::Value>)) {
    let (_, t) = text
        .iter_mut()
        .find(|(id, t)| {
            id.starts_with("openrouter.endpoints/") && t.matches("\"provider_name\"").count() > 1
        })
        .unwrap();
    let mut v: serde_json::Value = serde_json::from_str(t).unwrap();
    f(v.pointer_mut("/data/endpoints")
        .unwrap()
        .as_array_mut()
        .unwrap());
    *t = serde_json::to_string_pretty(&v).unwrap();
}

fn kimi(text: &mut BTreeMap<String, String>, row: &str) {
    replace(text, "fireworks.pricing", KIMI, row);
}

#[test]
fn unchanged_is_none() {
    let r = run(|_, _| {});
    assert_eq!(r.class, Class::None, "{r:?}");
    assert!(r.changes.is_empty() && r.reasons.is_empty());
}

#[test]
fn a_rate_within_2x_is_routine() {
    let r = run(|t, _| {
        kimi(
            t,
            r"| [Kimi K3](https://app.fireworks.ai/models/fireworks/kimi-k3) | \$3.30 / \$0.30 / \$15.00 | \$3.75 / \$0.375 / \$18.75 |",
        );
    });
    assert_eq!(r.class, Class::Routine, "{r:?}");
    // Fireworks prices no cache write, so the table bills it at the input rate: it moves too.
    let fields: Vec<&str> = r.changes.iter().map(|c| c.field.as_str()).collect();
    assert_eq!(
        fields,
        ["standard.cache_write_5m", "standard.input"],
        "{r:?}"
    );
    let c = &r.changes[1];
    assert_eq!(c.item, "card fireworks:accounts/fireworks/models/kimi-k3");
    assert_eq!(
        (c.old.as_deref(), c.new.as_deref()),
        (Some("$3/MTok"), Some("$3.3/MTok"))
    );
    assert_eq!(c.url, "https://docs.fireworks.ai/serverless/pricing.md");
    let md = r.markdown(&[], &[]);
    assert!(
        md.contains("| `standard.input` | $3/MTok | $3.3/MTok |"),
        "{md}"
    );
}

#[test]
fn exactly_2x_is_routine_and_more_is_review() {
    let row = |input: &str| {
        format!(
            r"| [Kimi K3](https://app.fireworks.ai/models/fireworks/kimi-k3) | \${input} / \$0.30 / \$15.00 | \$3.75 / \$0.375 / \$18.75 |"
        )
    };
    let r = run(|t, _| kimi(t, &row("6.00")));
    assert_eq!(r.class, Class::Routine, "{r:?}");
    let r = run(|t, _| kimi(t, &row("1.50")));
    assert_eq!(r.class, Class::Routine, "halving is 2x: {r:?}");
    let r = run(|t, _| kimi(t, &row("6.01")));
    assert_eq!(r.class, Class::Review, "{r:?}");
    assert_eq!(
        r.reasons,
        [
            "card fireworks:accounts/fireworks/models/kimi-k3: standard.cache_write_5m moved more than 2x",
            "card fireworks:accounts/fireworks/models/kimi-k3: standard.input moved more than 2x"
        ]
    );
    let r = run(|t, _| kimi(t, &row("1.40")));
    assert_eq!(r.class, Class::Review, "{r:?}");
    assert!(r.markdown(&[], &[]).starts_with("**Needs review.**"));
}

#[test]
fn a_rate_to_zero_is_review() {
    let r = run(|t, _| {
        kimi(
            t,
            r"| [Kimi K3](https://app.fireworks.ai/models/fireworks/kimi-k3) | \$3.00 / \$0.30 / \$0.00 | \$3.75 / \$0.375 / \$18.75 |",
        );
    });
    assert_eq!(r.class, Class::Review, "{r:?}");
    assert_eq!(
        r.reasons,
        ["card fireworks:accounts/fireworks/models/kimi-k3: standard.output went to zero"]
    );
}

#[test]
fn a_tier_that_disappears_is_review() {
    let r = run(|t, _| {
        kimi(
            t,
            r"| [Kimi K3](https://app.fireworks.ai/models/fireworks/kimi-k3) | \$3.00 / \$0.30 / \$15.00 | — |",
        );
    });
    assert_eq!(r.class, Class::Review, "{r:?}");
    assert!(
        r.reasons
            .iter()
            .any(|x| x.ends_with("kimi-k3: fast.input removed")),
        "{r:?}"
    );
    assert!(r.changes.iter().any(|c| c.new.is_none()), "{r:?}");
}

#[test]
fn an_endpoint_removed_or_added_is_review() {
    let r = run(|t, _| {
        openrouter(t, |eps| {
            eps.remove(0);
        });
    });
    assert_eq!(r.class, Class::Review, "{r:?}");
    assert!(
        r.reasons.len() == 1 && r.reasons[0].starts_with("removed: OpenRouter "),
        "{r:?}"
    );

    let r = run(|t, _| {
        openrouter(t, |eps| {
            let mut e = eps[0].clone();
            e["tag"] = "newhost/fp8".into();
            eps.push(e);
        });
    });
    assert_eq!(r.class, Class::Review, "{r:?}");
    assert!(
        r.reasons.len() == 1
            && r.reasons[0].starts_with("added: OpenRouter ")
            && r.reasons[0].ends_with(" @ newhost/fp8"),
        "{r:?}"
    );
}

#[test]
fn a_non_rate_value_that_changes_is_review() {
    let r = run(|t, _| {
        openrouter(t, |eps| eps[0]["provider_name"] = "Renamed Host".into());
    });
    assert_eq!(r.class, Class::Review, "{r:?}");
    assert!(
        r.reasons.len() == 1
            && r.reasons[0].contains(": host changed: ")
            && r.reasons[0].ends_with(" → Renamed Host"),
        "{r:?}"
    );
    assert!(r.changes.is_empty(), "a host is not a rate: {r:?}");
}

fn broken(r: &Report, kind: &str, says: &str) {
    assert_eq!(r.class, Class::Broken, "{r:?}");
    assert_eq!(r.kind, Some(kind), "{r:?}");
    assert!(r.error.as_deref().unwrap().contains(says), "{r:?}");
    assert!(r.changes.is_empty() && r.reasons.is_empty());
    let md = r.markdown(&[], &[]);
    assert!(md.contains(&format!("(`{kind}`)")), "{md}");
}

#[test]
fn a_layout_change_is_broken() {
    let r = run(|t, _| {
        replace(
            t,
            "fireworks.pricing",
            "| Model | Standard | Priority |",
            "| Model | Standard | Fast |",
        );
    });
    broken(&r, "reader", "fireworks");
}

#[test]
fn a_missing_quote_is_broken() {
    let r = run(|t, _| {
        replace(
            t,
            "groq.prompt_caching",
            "There is a 50% discount for cached input tokens.",
            "There is a 40% discount for cached input tokens.",
        );
    });
    broken(&r, "quote", "groq.cached");
}

#[test]
fn a_cross_check_disagreement_is_broken() {
    // Anthropic's list moves; AWS's Global SKUs (which must equal it) do not.
    let r = run(|t, _| {
        let a = t.get_mut("anthropic.pricing").unwrap();
        let line = a
            .lines()
            .find(|l| l.starts_with("| Claude Haiku 4.5 ") && l.contains("$1.25 / MTok"))
            .unwrap()
            .to_owned();
        *a = a.replacen(&line, &line.replace("$5 / MTok", "$6 / MTok"), 1);
    });
    broken(&r, "crosscheck", "are not Anthropic's list");
}

#[test]
fn a_stale_override_is_broken() {
    // Together's models API re-fetched after the promotion's end date.
    let r = run(|_, e| e.get_mut("together.models").unwrap().fetched = "2099-01-01".into());
    broken(&r, "stale", "[[override]]");
}
