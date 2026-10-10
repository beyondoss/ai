//! What a re-fetch changed, and whether a human must look before it ships: the drift workflow's
//! decision (`rates-sync classify`), made here rather than in shell.
//!
//! - **routine**: only numeric rate values moved, each within 2× either way and none to or from
//!   zero, and nothing was added or removed. The workflow opens a `rates-routine` PR.
//! - **review**: everything parsed, but a routine rule failed. A `rates-review` PR, reasons first.
//! - **broken**: a fetch, a reader (a layout change), a quote, a cross-check, or a stale
//!   `[[override]]`/`[[conflict]]` failed. Nothing is regenerated; the workflow files an issue.
//!
//! Every strict reader, cross-check, quote and override check runs inside [`crate::build::build`],
//! so a new table that builds has passed them all, and one that does not is broken.

use crate::build::{Card, Table, Tr};
use crate::dec::Rate;
use crate::snapshot::Entry;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// The table did not change.
    None,
    Routine,
    Review,
    Broken,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::None => "none",
            Class::Routine => "routine",
            Class::Review => "review",
            Class::Broken => "broken",
        }
    }

    /// The process exit code `rates-sync classify` reports it with (1 is reserved for an error).
    pub fn exit_code(self) -> u8 {
        match self {
            Class::None => 0,
            Class::Routine => 2,
            Class::Review => 3,
            Class::Broken => 4,
        }
    }
}

/// One rate that moved, appeared, or disappeared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    /// `card anthropic:claude-opus-4-8`, `OpenRouter z-ai/glm-5.3 @ deepinfra/fp8`.
    pub item: String,
    /// `standard.input`, `web_search per call`.
    pub field: String,
    pub old: Option<String>,
    pub new: Option<String>,
    pub url: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub class: Class,
    /// For a broken run: `fetch`, `reader`, `quote`, `crosscheck` or `stale`.
    pub kind: Option<&'static str>,
    pub error: Option<String>,
    /// Why a run is review, not routine.
    pub reasons: Vec<String>,
    pub changes: Vec<Change>,
}

impl Report {
    pub fn broken(kind: &'static str, error: String) -> Report {
        Report {
            class: Class::Broken,
            kind: Some(kind),
            error: Some(error),
            reasons: Vec::new(),
            changes: Vec::new(),
        }
    }
}

/// Which failure a generation error is. Every check reports its own fixed phrase (`build.rs`);
/// the fixture tests pin each one, so a reworded message fails them rather than misfiling.
pub fn broken_kind(err: &str) -> &'static str {
    let stale = (err.contains("[[override]]") || err.contains("[[conflict]]"))
        && (err.contains("is stale") || err.contains("remove it"));
    if stale {
        "stale"
    } else if err.contains("the quote is no longer on") || err.contains("so its quote cannot") {
        "quote"
    } else if err.contains("disagree")
        || err.contains("are not Anthropic's list")
        || err.contains("are not Global x")
        || err.contains("[[prose]] xai.long_context says")
    {
        "crosscheck"
    } else {
        "reader"
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Val {
    /// Per million tokens.
    Rate(Rate),
    /// Micro-dollars per call.
    Fee(u64),
    /// Anything that is not a price: a threshold, a premium, a schedule, a host, a wiring.
    Other(String),
}

impl Val {
    fn amount(&self) -> Option<u128> {
        match self {
            Val::Rate(r) => Some(r.0),
            Val::Fee(m) => Some(u128::from(*m)),
            Val::Other(_) => None,
        }
    }

    fn show(&self) -> String {
        match self {
            Val::Rate(r) => format!("{r}/MTok"),
            // A micro-dollar is 1e6 of a Rate's 1e-12 USD units.
            Val::Fee(m) => format!("{}/call", Rate(u128::from(*m) * 1_000_000)),
            Val::Other(s) => s.clone(),
        }
    }
}

/// `item → field → value`, plus each item's source id.
type Flat = BTreeMap<String, (String, BTreeMap<String, Val>)>;

fn tr(m: &mut BTreeMap<String, Val>, tier: &str, t: &Tr) {
    let mut put = |f: &str, r: Rate| m.insert(format!("{tier}.{f}"), Val::Rate(r));
    put("input", t.input);
    put("output", t.output);
    put("cache_read", t.cache_read);
    put("cache_write_5m", t.cache_write_5m);
    if let Some(r) = t.cache_write_1h {
        put("cache_write_1h", r);
    }
}

fn card(m: &mut BTreeMap<String, Val>, c: &Card) {
    tr(m, "standard", &c.standard);
    for (tier, t) in [
        ("long", &c.long),
        ("fast", &c.fast),
        ("fast_long", &c.fast_long),
        ("ultrafast", &c.ultrafast),
        ("ultrafast_long", &c.ultrafast_long),
        ("flex", &c.flex),
        ("flex_long", &c.flex_long),
    ] {
        if let Some(t) = t {
            tr(m, tier, t);
        }
    }
    if let Some(o) = &c.off_peak {
        tr(m, "off_peak", &o.rates);
        m.insert(
            "off_peak schedule".into(),
            Val::Other(format!(
                "peak {:?} weekdays_only={} holidays={:?} covered {}..{}",
                o.peak, o.weekdays_only, o.holidays, o.covered_from, o.covered_until
            )),
        );
    }
    if let Some((n, inclusive)) = c.long_context {
        let op = if inclusive { ">=" } else { ">" };
        m.insert("long_context".into(), Val::Other(format!("{op}{n}")));
    }
    if let Some(g) = c.geo_us {
        m.insert("geo_us".into(), Val::Other(format!("{g} bps")));
    }
}

/// The source a card's numbers come from (`build.rs`, `Ctx::card`).
fn card_source(name: &str) -> &'static str {
    match name.split_once(':').map_or("", |(v, _)| v) {
        "anthropic" => "anthropic.pricing",
        "bedrock" => "aws.bedrock",
        "openai" => "openai.pricing",
        "xai" => "xai.language_models",
        "deepseek" => "deepseek.pricing",
        "together" => "together.models",
        "fireworks" => "fireworks.pricing",
        "groq" => "groq.models",
        _ => "",
    }
}

fn flatten(t: &Table) -> Flat {
    let mut f = Flat::new();
    let fees: BTreeMap<&str, &[(String, u64)]> = t
        .tools
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    f.insert(
        "OpenRouter credit fee".into(),
        (
            "openrouter.pricing".into(),
            BTreeMap::from([(
                "multiplier".into(),
                Val::Other(format!("{} bps", t.credit_fee_bps)),
            )]),
        ),
    );
    // The OR_SEARCH_* lists are named after an endpoint's fee; that fee is on its endpoint below.
    for (name, list) in t.tools.iter().filter(|(n, _)| !n.starts_with("OR_SEARCH_")) {
        let m = list
            .iter()
            .map(|(tool, micros)| (tool.clone(), Val::Fee(*micros)))
            .collect();
        f.insert(format!("tool fees {name}"), (String::new(), m));
    }
    for (name, c) in &t.cards {
        let mut m = BTreeMap::new();
        card(&mut m, c);
        if let Some(tools) = &c.tools {
            m.insert("tools".into(), Val::Other(tools.clone()));
        }
        f.insert(format!("card {name}"), (card_source(name).into(), m));
    }
    for (slug, eps) in &t.openrouter {
        for e in eps {
            let mut m = BTreeMap::new();
            card(&mut m, &e.card);
            m.insert("host".into(), Val::Other(e.host.clone()));
            m.insert("class".into(), Val::Other(e.class.into()));
            if let Some(name) = &e.card.tools {
                let v = match fees.get(name.as_str()).and_then(|l| l.first()) {
                    Some((_, micros)) => Val::Fee(*micros),
                    None => Val::Other(name.clone()),
                };
                m.insert("web_search per call".into(), v);
            }
            f.insert(
                format!("OpenRouter {slug} @ {}", e.tag),
                (format!("openrouter.endpoints/{slug}"), m),
            );
        }
    }
    for r in &t.rows {
        let mut m = BTreeMap::new();
        m.insert("list".into(), Val::Other(format!("{:?}", r.list)));
        for (p, c) in &r.cost {
            m.insert(format!("cost {p}"), Val::Other(format!("{c:?}")));
        }
        f.insert(format!("row {}", r.model), (String::new(), m));
    }
    f
}

/// Compare the committed table with the one the re-fetched snapshots generate (or the error they
/// failed with). `entries` (the new manifest) gives each item's source URL.
pub fn classify(
    old: &Table,
    new: std::result::Result<&Table, &str>,
    entries: &BTreeMap<String, Entry>,
) -> Report {
    let new = match new {
        Ok(t) => t,
        Err(e) => return Report::broken(broken_kind(e), e.to_owned()),
    };
    let (a, b) = (flatten(old), flatten(new));
    let mut reasons = Vec::new();
    let mut changes = Vec::new();
    for item in b.keys().filter(|k| !a.contains_key(*k)) {
        reasons.push(format!("added: {item}"));
    }
    for (item, (src, am)) in &a {
        let Some((_, bm)) = b.get(item) else {
            reasons.push(format!("removed: {item}"));
            continue;
        };
        let url = entries.get(src).map_or_else(String::new, |e| e.url.clone());
        let fields = am.keys().chain(bm.keys().filter(|k| !am.contains_key(*k)));
        for field in fields {
            let (x, y) = (am.get(field), bm.get(field));
            if x == y {
                continue;
            }
            let priced = x.or(y).is_some_and(|v| v.amount().is_some());
            if !priced {
                let show = |v: Option<&Val>| v.map_or("absent".into(), Val::show);
                reasons.push(format!(
                    "{item}: {field} changed: {} → {}",
                    show(x),
                    show(y)
                ));
                continue;
            }
            match (x.and_then(Val::amount), y.and_then(Val::amount)) {
                (Some(o), Some(n)) => {
                    if o == 0 || n == 0 {
                        reasons.push(format!(
                            "{item}: {field} went {} zero",
                            if n == 0 { "to" } else { "from" }
                        ));
                    } else if n > o.saturating_mul(2) || o > n.saturating_mul(2) {
                        reasons.push(format!("{item}: {field} moved more than 2x"));
                    }
                }
                (o, _) => reasons.push(format!(
                    "{item}: {field} {}",
                    if o.is_some() { "removed" } else { "added" }
                )),
            }
            changes.push(Change {
                item: item.clone(),
                field: field.clone(),
                old: x.map(Val::show),
                new: y.map(Val::show),
                url: url.clone(),
            });
        }
    }
    let class = if reasons.is_empty() && changes.is_empty() {
        Class::None
    } else if reasons.is_empty() {
        Class::Routine
    } else {
        Class::Review
    };
    Report {
        class,
        kind: None,
        error: None,
        reasons,
        changes,
    }
}

impl Report {
    /// The PR body (routine, review) or the issue body (broken).
    pub fn markdown(&self, sources_changed: &[String], skipped: &[String]) -> String {
        let mut s = String::new();
        match self.class {
            Class::None => s.push_str("No vendor rate changed.\n"),
            Class::Broken => {
                let _ = write!(
                    s,
                    "The daily rate sync failed (`{}`), so nothing was regenerated. The committed \
                     table still prices every row; fix the source reader, the quote, or the rule in \
                     `verify/catalog_truth.toml`, then run `mise run rates:sync`.\n\n```text\n{}\n```\n",
                    self.kind.unwrap_or("unknown"),
                    self.error.as_deref().unwrap_or("").trim_end()
                );
            }
            Class::Routine | Class::Review => {
                if self.class == Class::Review {
                    s.push_str("**Needs review.** A routine rule failed:\n\n");
                    for r in &self.reasons {
                        let _ = writeln!(s, "- {r}");
                    }
                    s.push('\n');
                } else {
                    s.push_str(
                        "**Routine.** Only rate values moved, each within 2x, none to or from \
                         zero; every reader, cross-check and quote passed.\n\n",
                    );
                }
                if !self.changes.is_empty() {
                    s.push_str(
                        "| Item | Rate | Old | New | Source |\n| --- | --- | --- | --- | --- |\n",
                    );
                    for c in &self.changes {
                        let _ = writeln!(
                            s,
                            "| {} | `{}` | {} | {} | {} |",
                            c.item,
                            c.field,
                            c.old.as_deref().unwrap_or("—"),
                            c.new.as_deref().unwrap_or("—"),
                            if c.url.is_empty() { "—" } else { &c.url }
                        );
                    }
                    s.push('\n');
                }
                s.push_str(
                    "`RATE_VERSION` is bumped (`rates-sync rate-version`). A golden vector in \
                     `verify/pricing_vectors.json` that priced a moved rate fails CI until it is \
                     recomputed. A human merges this PR.\n",
                );
            }
        }
        if !sources_changed.is_empty() {
            let _ = write!(
                s,
                "\nSnapshots that changed: {}\n",
                sources_changed
                    .iter()
                    .map(|x| format!("`{x}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if !skipped.is_empty() {
            let _ = writeln!(
                s,
                "\nNot re-fetched (their keys are not set): {}",
                skipped
                    .iter()
                    .map(|x| format!("`{x}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        s
    }

    pub fn to_json(&self, sources_changed: &[String], skipped: &[String]) -> Value {
        json!({
            "class": self.class.as_str(),
            "kind": self.kind,
            "error": self.error,
            "reasons": self.reasons,
            "changes": self.changes.iter().map(|c| json!({
                "item": c.item,
                "field": c.field,
                "old": c.old,
                "new": c.new,
                "url": c.url,
            })).collect::<Vec<_>>(),
            "sources_changed": sources_changed,
            "skipped": skipped,
            "markdown": self.markdown(sources_changed, skipped),
        })
    }
}
