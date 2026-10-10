//! `rates-sync catalog-drift`: new models and retirements against the catalog, found and
//! classified by rule, with the edit for each one a rule can make.
//!
//! [`detect`] is a pure function of [`Inputs`] (the vendors' listings, deprecation pages and the
//! freshly fetched rate snapshots, saved under one directory) and the catalog this binary was
//! compiled with, so the same inputs always give the same findings, byte for byte. Each finding
//! is one of:
//!
//! - **successor**: a newer version in a line of models a row carries, served by (a subset of)
//!   the predecessor's hosts, in the same feature class as far as the sources show. Its row is
//!   generated from the predecessor's: each host's id as that host lists it, the predecessor's
//!   order and paths, card limits and capability bits, and prices from cards the rate generator
//!   builds from the snapshots.
//! - **retirement**: a candidate a vendor retires within [`RETIRE_AHEAD_DAYS`] (or has stopped
//!   listing) leaves its row's fallback list; a row whose every candidate retires is removed and
//!   its name becomes an alias of its successor row.
//! - **needs-human**: everything else, each with the reasons no rule applies.
//!
//! See `crates/providers/ARCHITECTURE.md`, "Catalog maintenance".

use crate::build::{self, Card, Endpoint};
use crate::catalog_edit::{self, Addition, Edit, NewCand, Removal};
use crate::lineage::{self, brand, parse, undated, with_version};
use crate::snapshot::Store;
use crate::sources;
use crate::spec::{CardSpec, Spec};
use crate::{Result, fetch};
use providers::catalog::{self, ModelRoute};
use providers::{ProviderId, by_id};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// A scheduled retirement is acted on this many days ahead of its date: the window in which
/// `no_catalog_row_outlives_its_retirement` starts warning.
pub const RETIRE_AHEAD_DAYS: u32 = 14;
/// More candidates than this vanishing from one host's listing in one run reads as a broken
/// listing, not as retirements: they are reported for a human instead of removed.
pub const VANISH_LIMIT: usize = 2;

pub const ANTHROPIC_MODELS: &str = "https://api.anthropic.com/v1/models?limit=1000";
pub const OPENAI_MODELS: &str = "https://api.openai.com/v1/models";
pub const BEDROCK_PROFILES: &str = "https://bedrock.us-east-1.amazonaws.com/inference-profiles?maxResults=1000&typeEquals=SYSTEM_DEFINED";
/// Where each vendor publishes its retirements as markdown tables of (date, model).
pub const DEPRECATION_PAGES: &[(&str, &str)] = &[
    (
        "anthropic",
        "https://platform.claude.com/docs/en/about-claude/model-deprecations.md",
    ),
    (
        "openai",
        "https://developers.openai.com/api/docs/deprecations.md",
    ),
    ("together", "https://docs.together.ai/docs/deprecations.md"),
];

/// Everything [`detect`] reads. Saved and loaded as one directory, so a run can be replayed.
pub struct Inputs {
    /// `YYYY-MM-DD`, UTC.
    pub today: String,
    /// The rate snapshots, freshly fetched (a scratch copy of `verify/rates_sources/`).
    pub store: Store,
    /// `GET https://openrouter.ai/api/v1/models?output_modalities=all`, as served.
    pub openrouter: String,
    /// `GET /api/v1/models/{slug}/endpoints`, as served, for each slug no row prices yet.
    pub endpoints: BTreeMap<String, String>,
    /// The vendors' own `/v1/models` (Anthropic, OpenAI) and Bedrock's inference profiles, as
    /// served; `None` when the key was not set.
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    pub bedrock: Option<String>,
    /// Provider → its deprecation page, as served.
    pub deprecations: BTreeMap<String, String>,
}

/// The catalog the findings are about.
pub struct Catalog<'a> {
    pub rows: &'a [ModelRoute],
    pub spec: &'a Spec,
    /// `verify/catalog_truth.toml`, parsed.
    pub truth: &'a toml::Table,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Successor,
    Retirement,
    NeedsHuman,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Successor => "successor",
            Class::Retirement => "retirement",
            Class::NeedsHuman => "needs-human",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Finding {
    /// Stable id: the PR branch is `catalog/{slug}`.
    pub slug: String,
    pub class: Class,
    /// The row added or retiring (or the vendor id, for a model no row's line holds).
    pub model: String,
    pub vendor: String,
    pub title: String,
    /// List price (input, output, cache read, cache write), USD per million tokens.
    pub prices: Option<[String; 4]>,
    pub sources: Vec<String>,
    /// What was found.
    pub notes: Vec<String>,
    /// Why no rule applies (needs-human only).
    pub reasons: Vec<String>,
    pub edit: Option<Edit>,
    /// Gateway providers of the row after the edit: what the live test must reach.
    pub providers: Vec<String>,
}

// ---------------------------------------------------------------------------------------------
// Listings
// ---------------------------------------------------------------------------------------------

/// One model a host lists.
#[derive(Clone, Debug, Default)]
struct Listed {
    id: String,
    aliases: Vec<String>,
    created: Option<u64>,
    /// The host's own scheduled retirement: OpenRouter `expiration_date`, Anthropic
    /// `retires_at`, OpenAI `shutdown_date`.
    retires: Option<String>,
}

/// OpenRouter's description of one model: what the feature-class check compares.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct OrCard {
    input: Vec<String>,
    /// Of [`COMPARED_PARAMS`], the ones it supports.
    params: Vec<String>,
    context: Option<u64>,
    max_output: Option<u64>,
    /// (prompt, completion) per token, as listed, for display.
    pricing: Option<(String, String)>,
}

const COMPARED_PARAMS: &[&str] = &[
    "tools",
    "tool_choice",
    "structured_outputs",
    "response_format",
    "reasoning",
];

struct Hosts {
    /// Gateway provider name → what it lists.
    lists: BTreeMap<&'static str, Vec<Listed>>,
    /// Where each host's listing came from.
    urls: BTreeMap<&'static str, &'static str>,
    or_cards: BTreeMap<String, OrCard>,
}

/// RFC 3339 → Unix seconds.
fn rfc3339(s: &str) -> Option<u64> {
    let day = u64::from(build::day(s.get(..10)?).ok()?);
    let t = s.get(11..19)?;
    let h: u64 = t.get(..2)?.parse().ok()?;
    let m: u64 = t.get(3..5)?.parse().ok()?;
    let sec: u64 = t.get(6..8)?.parse().ok()?;
    Some(day * 86_400 + h * 3600 + m * 60 + sec)
}

fn date_of_value(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if s.len() >= 10 && build::day(&s[..10]).is_ok() => {
            Some(s[..10].to_owned())
        }
        Value::Number(n) => n.as_i64().map(fetch::date_of),
        _ => None,
    }
}

fn json(text: &str, what: &str) -> Result<Value> {
    serde_json::from_str(text).map_err(|e| format!("{what}: not JSON: {e}"))
}

fn items<'a>(v: &'a Value, key: &str, what: &str) -> Result<&'a Vec<Value>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{what}: no `{key}` array"))
}

impl Hosts {
    fn read(inputs: &Inputs) -> Result<Hosts> {
        let mut lists = BTreeMap::new();
        let mut urls = BTreeMap::new();
        let mut or_cards = BTreeMap::new();
        let or = json(&inputs.openrouter, "openrouter models")?;
        let mut v = Vec::new();
        for m in items(&or, "data", "openrouter models")? {
            let Some(id) = m["id"].as_str() else { continue };
            let strs = |p: &str| -> Vec<String> {
                let mut s: Vec<String> = m
                    .pointer(p)
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_owned))
                            .collect()
                    })
                    .unwrap_or_default();
                s.sort();
                s
            };
            let params = strs("/supported_parameters");
            or_cards.insert(
                id.to_owned(),
                OrCard {
                    input: strs("/architecture/input_modalities"),
                    params: COMPARED_PARAMS
                        .iter()
                        .filter(|p| params.iter().any(|x| x == *p))
                        .map(|p| (*p).to_owned())
                        .collect(),
                    context: m["context_length"].as_u64(),
                    max_output: m
                        .pointer("/top_provider/max_completion_tokens")
                        .and_then(Value::as_u64),
                    pricing: m["pricing"]["prompt"]
                        .as_str()
                        .zip(m["pricing"]["completion"].as_str())
                        .map(|(a, b)| (a.to_owned(), b.to_owned())),
                },
            );
            v.push(Listed {
                id: id.to_owned(),
                aliases: Vec::new(),
                created: m["created"].as_u64(),
                retires: date_of_value(&m["expiration_date"]),
            });
        }
        lists.insert("openrouter", v);
        urls.insert("openrouter", sources::OPENROUTER_MODELS);
        if let Some(a) = &inputs.anthropic {
            let a = json(a, "anthropic models")?;
            let v = items(&a, "data", "anthropic models")?
                .iter()
                .filter_map(|m| {
                    Some(Listed {
                        id: m["id"].as_str()?.to_owned(),
                        aliases: Vec::new(),
                        created: m["created_at"].as_str().and_then(rfc3339),
                        retires: date_of_value(&m["retires_at"]),
                    })
                })
                .collect();
            lists.insert("anthropic", v);
            urls.insert("anthropic", ANTHROPIC_MODELS);
        }
        if let Some(o) = &inputs.openai {
            let o = json(o, "openai models")?;
            let v = items(&o, "data", "openai models")?
                .iter()
                .filter_map(|m| {
                    Some(Listed {
                        id: m["id"].as_str()?.to_owned(),
                        aliases: Vec::new(),
                        created: m["created"].as_u64(),
                        retires: date_of_value(&m["shutdown_date"]),
                    })
                })
                .collect();
            lists.insert("openai", v);
            urls.insert("openai", OPENAI_MODELS);
        }
        if let Some(b) = &inputs.bedrock {
            let b = json(b, "bedrock inference profiles")?;
            let v = items(
                &b,
                "inferenceProfileSummaries",
                "bedrock inference profiles",
            )?
            .iter()
            .filter(|p| p["status"] == "ACTIVE")
            .filter_map(|p| {
                Some(Listed {
                    id: p["inferenceProfileId"].as_str()?.to_owned(),
                    ..Listed::default()
                })
            })
            .collect();
            lists.insert("bedrock", v);
            urls.insert("bedrock", BEDROCK_PROFILES);
        }
        if let Ok(x) = inputs.store.get("xai.language_models") {
            let x = json(x, "xai language models")?;
            let v = items(&x, "models", "xai language models")?
                .iter()
                .filter_map(|m| {
                    Some(Listed {
                        id: m["id"].as_str()?.to_owned(),
                        aliases: m["aliases"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|s| s.as_str().map(str::to_owned))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        created: m["created"].as_u64(),
                        retires: None,
                    })
                })
                .collect();
            lists.insert("xai", v);
            urls.insert("xai", sources::XAI_MODELS);
        }
        if let Ok(t) = inputs.store.get("together.models") {
            let t = json(t, "together models")?;
            let v = t
                .as_array()
                .ok_or("together models: not an array")?
                .iter()
                .filter_map(|m| {
                    Some(Listed {
                        id: m["id"].as_str()?.to_owned(),
                        ..Listed::default()
                    })
                })
                .collect();
            lists.insert("together", v);
            urls.insert("together", sources::TOGETHER_MODELS);
        }
        Ok(Hosts {
            lists,
            urls,
            or_cards,
        })
    }

    /// OpenRouter's listed (prompt, completion) price, per million tokens, for display.
    fn or_price(&self, slug: &str) -> Option<[String; 4]> {
        let (p, c) = self.or_cards.get(slug)?.pricing.as_ref()?;
        let pm = |s: &str| crate::dec::Rate::per_token(s).ok()?.table_text().ok();
        Some([pm(p)?, pm(c)?, String::new(), String::new()])
    }

    /// The listing entry for a candidate id: by id or alias, or by a dated snapshot of it
    /// (Anthropic lists `claude-haiku-4-5-20251001` for `claude-haiku-4-5`).
    fn find(&self, host: &str, id: &str) -> Option<&Listed> {
        let list = self.lists.get(host)?;
        list.iter()
            .find(|m| m.id.eq_ignore_ascii_case(id) || m.aliases.iter().any(|a| a == id))
            .or_else(|| {
                list.iter()
                    .find(|m| undated(&m.id).eq_ignore_ascii_case(id))
            })
    }
}

/// What a host lists for version `new` of the line `pred` (an id at that host) is in.
#[derive(Debug, PartialEq, Eq)]
enum Derived {
    One(String),
    NotListed,
    Ambiguous(Vec<String>),
    NoLineage,
}

fn derive(list: &[Listed], pred: &str, new: &[u32]) -> Derived {
    let Some(p) = parse(pred) else {
        return Derived::NoLineage;
    };
    let mut hits: Vec<&str> = list
        .iter()
        .flat_map(|m| std::iter::once(&m.id).chain(&m.aliases))
        .map(String::as_str)
        .filter(|id| parse(id).is_some_and(|l| l.line == p.line && l.version == new))
        .collect();
    hits.sort_unstable();
    hits.dedup();
    let pick = |v: Vec<&str>| match v[..] {
        [one] => Some(one.to_owned()),
        _ => None,
    };
    if hits.is_empty() {
        return Derived::NotListed;
    }
    if let Some(one) = pick(hits.clone()) {
        // An undated predecessor keeps the undated spelling (Anthropic's alias of a dated snapshot).
        return Derived::One(if p.dated { one } else { undated(&one) });
    }
    let same: Vec<&str> = hits
        .iter()
        .copied()
        .filter(|id| parse(id).is_some_and(|l| l.dated == p.dated))
        .collect();
    if let Some(one) = pick(same) {
        return Derived::One(one);
    }
    let forms: BTreeSet<String> = hits.iter().map(|h| undated(h)).collect();
    if !p.dated && forms.len() == 1 {
        return Derived::One(forms.into_iter().next().unwrap_or_default());
    }
    Derived::Ambiguous(hits.into_iter().map(str::to_owned).collect())
}

/// One id across hosts: lowercase, no vendor prefix, no snapshot, `.` as `-`.
fn norm(id: &str) -> String {
    let name = id.rsplit('/').next().unwrap_or(id);
    undated(name).to_ascii_lowercase().replace(['.', '_'], "-")
}

// ---------------------------------------------------------------------------------------------
// The truth file
// ---------------------------------------------------------------------------------------------

fn truth_list<'a>(t: &'a toml::Table, k: &str) -> &'a [toml::Value] {
    t.get(k)
        .and_then(toml::Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn tfield<'a>(e: &'a toml::Value, k: &str) -> Option<&'a str> {
    e.get(k).and_then(toml::Value::as_str)
}

impl Catalog<'_> {
    fn truth_row(&self, model: &str) -> Option<&toml::Value> {
        truth_list(self.truth, "row")
            .iter()
            .find(|e| tfield(e, "model") == Some(model))
    }

    fn pricing(&self, model: &str) -> Option<&crate::spec::Pricing> {
        self.spec.pricing.iter().find(|p| p.model == model)
    }

    fn card_spec(&self, name: &str) -> Option<&CardSpec> {
        self.spec.cards.iter().find(|c| c.name == name)
    }

    /// A `[[not_carried]]`, `[[retired]]` or `[[not_serverless]]` entry naming (host, id), or
    /// `model` naming it.
    fn recorded(&self, host: &str, id: &str) -> bool {
        ["not_carried", "retired", "not_serverless"]
            .iter()
            .any(|list| {
                truth_list(self.truth, list).iter().any(|e| {
                    (tfield(e, "provider") == Some(host)
                        && tfield(e, "id")
                            .is_some_and(|x| x.eq_ignore_ascii_case(id) || undated(id) == x))
                        || tfield(e, "model") == Some(id)
                })
            })
    }

    fn retired_entry(&self, host: &str, id: &str) -> bool {
        truth_list(self.truth, "retired")
            .iter()
            .any(|e| tfield(e, "provider") == Some(host) && tfield(e, "id") == Some(id))
    }

    /// Entries in the truth file's per-candidate measurements that name `needle`.
    fn measured(&self, needles: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        for list in [
            "schema_unenforced",
            "conflict",
            "override",
            "superseded",
            "promo",
        ] {
            for e in truth_list(self.truth, list) {
                let hit = e.as_table().is_some_and(|t| {
                    t.values()
                        .any(|v| v.as_str().is_some_and(|s| needles.contains(&s)))
                });
                if hit {
                    out.push(format!("[[{list}]]"));
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// A row, or a retired row's alias, already has this name.
    fn exists(&self, name: &str) -> bool {
        self.rows.iter().any(|r| r.model.eq_ignore_ascii_case(name))
            || catalog::ALIASES
                .iter()
                .any(|(a, _)| a.eq_ignore_ascii_case(name))
    }
}

/// The candidates and Responses arms of a row, each (host, id, path).
fn arms(row: &ModelRoute) -> Vec<(bool, &catalog::Candidate)> {
    row.candidates
        .iter()
        .map(|c| (false, c))
        .chain(row.responses.iter().map(|c| (true, c)))
        .collect()
}

fn host(c: &catalog::Candidate) -> &'static str {
    by_id(c.provider).name
}

/// A per-candidate rule `catalog.rs` keeps by hand for this candidate.
fn hand_rule(c: &catalog::Candidate) -> Option<&'static str> {
    if catalog::stream_only(c) {
        Some("stream-only (STREAM_ONLY)")
    } else if catalog::tool_thinking(c) != catalog::ToolThinking::Free {
        Some("thinking off with tools (TOOL_THINKING)")
    } else if !catalog::serves_file_input(c) {
        Some("refuses file input (REFUSES_FILE_INPUT)")
    } else if c.provider != ProviderId::Bedrock && !catalog::serves_structured_outputs(c) {
        Some("does not hold a JSON schema (REFUSES_STRUCTURED_OUTPUTS)")
    } else if catalog::serves_fast_mode(c) {
        Some("serves fast mode (FAST_MODE_MODELS)")
    } else {
        None
    }
}

fn row_line(model: &str) -> Option<(String, Vec<u32>)> {
    parse(model).map(|l| (l.line, l.version))
}

fn slug(prefix: &str, s: &str) -> String {
    let mut out = String::from(prefix);
    for c in s.chars() {
        out.push(if c.is_ascii_alphanumeric() || c == '.' {
            c.to_ascii_lowercase()
        } else {
            '-'
        });
    }
    out
}

fn add_days(date: &str, days: u32) -> Result<String> {
    Ok(fetch::date_of(i64::from(build::day(date)? + days) * 86_400))
}

fn list_price(card: &Card) -> Result<[String; 4]> {
    let s = card.standard;
    Ok([
        s.input.table_text()?,
        s.output.table_text()?,
        s.cache_read.table_text()?,
        s.cache_write_5m.table_text()?,
    ])
}

/// What sets a card's pricing class apart: its tiers, not its rates.
fn tiers(c: &Card) -> String {
    let mut v = Vec::new();
    match c.long_context {
        Some((n, true)) => v.push(format!("long context >={n}")),
        Some((n, false)) => v.push(format!("long context >{n}")),
        None => {}
    }
    for (on, name) in [
        (c.fast.is_some(), "fast"),
        (c.ultrafast.is_some(), "ultrafast"),
        (c.flex.is_some(), "flex"),
        (c.off_peak.is_some(), "off-peak"),
        (c.geo_us.is_some(), "US-only premium"),
    ] {
        if on {
            v.push(name.to_owned());
        }
    }
    if let Some(t) = &c.tools {
        v.push(format!("tool fees {t}"));
    }
    if v.is_empty() {
        "standard only".into()
    } else {
        v.join(", ")
    }
}

fn or_tiers(eps: &[Endpoint]) -> String {
    let mut longs: Vec<String> = eps
        .iter()
        .filter_map(|e| e.card.long_context.map(|(n, _)| format!(">{n}")))
        .collect();
    longs.sort();
    longs.dedup();
    if longs.is_empty() {
        "no long-context tier".into()
    } else {
        format!("long context {}", longs.join("/"))
    }
}

fn price_of(p: &providers::catalog::ListPrice) -> [String; 4] {
    [
        p.input.to_owned(),
        p.output.to_owned(),
        p.cache_read.to_owned(),
        p.cache_write.to_owned(),
    ]
}

// ---------------------------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------------------------

/// The OpenRouter slugs new models would be priced from: [`Inputs::endpoints`] should hold each.
pub fn wanted_endpoints(inputs: &Inputs, cat: &Catalog<'_>) -> Result<Vec<String>> {
    let hosts = Hosts::read(inputs)?;
    let mut out = Vec::new();
    for g in discover(&hosts, cat).news {
        let pred = &cat.rows[g.pred];
        for (_, c) in arms(pred) {
            if c.provider == ProviderId::OpenRouter
                && let Some(list) = hosts.lists.get("openrouter")
                && let Derived::One(id) = derive(list, c.upstream_model, &g.version)
            {
                out.push(id);
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// A newer version of a line of rows, seen at one or more hosts.
struct NewGroup {
    pred: usize,
    version: Vec<u32>,
    seen: Vec<(&'static str, String)>,
}

/// A candidate in a host's line: (row, version, its release time at that host).
type Carried = (usize, Vec<u32>, Option<u64>);

struct Discovery {
    news: Vec<NewGroup>,
    /// (host, id, reason).
    humans: Vec<(&'static str, String, String)>,
}

fn discover(hosts: &Hosts, cat: &Catalog<'_>) -> Discovery {
    let mut groups: BTreeMap<(String, Vec<u32>), NewGroup> = BTreeMap::new();
    let mut humans: BTreeMap<String, (&'static str, String, String)> = BTreeMap::new();
    for (&h, list) in &hosts.lists {
        // The lines this host's candidates are in: line → (row, version, created at this host).
        let mut lines: BTreeMap<String, Vec<Carried>> = BTreeMap::new();
        let mut carried: BTreeSet<String> = BTreeSet::new();
        for (ri, row) in cat.rows.iter().enumerate() {
            for (_, c) in arms(row) {
                if host(c) != h {
                    continue;
                }
                carried.insert(c.upstream_model.to_ascii_lowercase());
                if let Some(l) = parse(c.upstream_model) {
                    let created = hosts.find(h, c.upstream_model).and_then(|m| m.created);
                    lines
                        .entry(l.line)
                        .or_default()
                        .push((ri, l.version, created));
                }
            }
        }
        // OpenRouter is where a new family shows: its brands, and each brand's newest release.
        let mut brands: BTreeMap<String, u64> = BTreeMap::new();
        if h == "openrouter" {
            for id in &carried {
                if let Some(c) = hosts.find(h, id).and_then(|m| m.created) {
                    let e = brands.entry(brand(id)).or_default();
                    *e = (*e).max(c);
                }
            }
        }
        for m in list {
            let id = m.id.as_str();
            if h == "openrouter" && (id.contains(':') || id.starts_with('~')) {
                continue;
            }
            let lower = id.to_ascii_lowercase();
            if carried.contains(&lower)
                || carried.contains(&undated(&lower))
                || m.aliases
                    .iter()
                    .any(|a| carried.contains(&a.to_ascii_lowercase()))
                || cat.recorded(h, id)
            {
                continue;
            }
            let lin = parse(id);
            let hit = lin
                .as_ref()
                .and_then(|l| lines.get(&l.line).map(|c| (l, c)));
            if let Some((l, carried_line)) = hit {
                if carried_line.iter().any(|(_, v, _)| *v == l.version) {
                    continue;
                }
                let Some((top_row, top_v, _)) = carried_line.iter().max_by(|a, b| a.1.cmp(&b.1))
                else {
                    continue;
                };
                let newest = carried_line.iter().filter_map(|x| x.2).max();
                let Some((rl, _)) = row_line(cat.rows[*top_row].model) else {
                    continue;
                };
                // The row's own line decides the predecessor: its highest version, any host.
                let pred = cat
                    .rows
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| {
                        row_line(r.model)
                            .filter(|(x, _)| *x == rl)
                            .map(|(_, v)| (i, v))
                    })
                    .max_by(|a, b| a.1.cmp(&b.1));
                let Some((pred, pred_v)) = pred else { continue };
                let at_row = cat
                    .rows
                    .iter()
                    .any(|r| row_line(r.model).is_some_and(|(x, v)| x == rl && v == l.version));
                if at_row {
                    // The row exists; this host is one the row does not use.
                    continue;
                }
                if l.version > *top_v && l.version > pred_v {
                    groups
                        .entry((rl, l.version.clone()))
                        .or_insert_with(|| NewGroup {
                            pred,
                            version: l.version.clone(),
                            seen: Vec::new(),
                        })
                        .seen
                        .push((h, id.to_owned()));
                } else if m.created.zip(newest).is_some_and(|(c, n)| c > n) {
                    humans.entry(format!("{rl}@{:?}", l.version)).or_insert((
                        h,
                        id.to_owned(),
                        format!(
                            "ambiguous lineage: released after {}, but its version {} does not follow {}",
                            cat.rows[*top_row].model,
                            lineage::format_version(&l.version, '.'),
                            lineage::format_version(top_v, '.'),
                        ),
                    ));
                }
            } else if h == "openrouter"
                && let Some(newest) = brands.get(&brand(id))
                && m.created.is_some_and(|c| c > *newest)
            {
                humans.entry(norm(id)).or_insert((
                    h,
                    id.to_owned(),
                    format!(
                        "new family: no catalog row is in its line ({}), and it was released after every {} model the catalog carries",
                        lin.map_or_else(|| id.to_owned(), |l| l.line),
                        brand(id)
                    ),
                ));
            }
        }
    }
    Discovery {
        news: groups.into_values().collect(),
        humans: humans.into_values().collect(),
    }
}

/// Every finding, sorted by slug.
pub fn detect(inputs: &Inputs, cat: &Catalog<'_>) -> Result<Vec<Finding>> {
    let hosts = Hosts::read(inputs)?;
    let d = discover(&hosts, cat);
    let mut out = Vec::new();
    for g in &d.news {
        if let Some(f) = successor(inputs, &hosts, cat, g)? {
            out.push(f);
        }
    }
    for (h, id, reason) in d.humans {
        out.push(Finding {
            slug: slug("review-", &format!("{h}-{id}")),
            class: Class::NeedsHuman,
            model: id.clone(),
            vendor: id.split_once('/').map_or(h, |(v, _)| v).to_owned(),
            title: format!("{id} ({h})"),
            prices: hosts.or_price(&id),
            sources: vec![hosts.urls.get(h).copied().unwrap_or_default().to_owned()],
            notes: vec![format!("{h} lists {id}")],
            reasons: vec![reason],
            edit: None,
            providers: Vec::new(),
        });
    }
    out.extend(retirements(inputs, &hosts, cat)?);
    out.sort_by(|a, b| a.slug.cmp(&b.slug));
    out.dedup_by(|a, b| a.slug == b.slug);
    Ok(out)
}

/// A temporary store that also holds what pricing a new OpenRouter slug needs.
fn with_endpoints(inputs: &Inputs, cat: &Catalog<'_>, slugs: &[String]) -> Result<Store> {
    let mut text = inputs.store.text.clone();
    let mut all = cat.spec.openrouter_slugs();
    all.extend(slugs.iter().cloned());
    all.sort();
    all.dedup();
    for d in sources::all(&all) {
        if d.id == "openrouter.models" {
            let t = sources::normalize(&d, inputs.openrouter.as_bytes(), &all)?;
            text.insert(d.id, t);
        } else if let Some(slug) = d.id.strip_prefix("openrouter.endpoints/")
            && !text.contains_key(&d.id)
        {
            let raw = inputs
                .endpoints
                .get(slug)
                .ok_or_else(|| format!("no endpoint list fetched for {slug}"))?;
            let t = sources::normalize(&d, raw.as_bytes(), &all)?;
            text.insert(d.id, t);
        }
    }
    Ok(Store {
        dir: inputs.store.dir.clone(),
        entries: inputs.store.entries.clone(),
        text,
    })
}

/// The source ids a vendor's card reads.
fn card_sources(vendor: &str) -> &'static [&'static str] {
    match vendor {
        "anthropic" => &["anthropic.pricing"],
        "bedrock" => &["aws.bedrock", "anthropic.pricing"],
        "openai" => &["openai.pricing"],
        "xai" => &["xai.language_models", "xai.pricing"],
        "deepseek" => &["deepseek.pricing", "gov_cn.holidays"],
        "together" => &["together.models", "together.docs", "together.pricing"],
        "groq" => &["groq.models", "groq.prompt_caching", "groq.flex"],
        "fireworks" => &["fireworks.pricing"],
        _ => &[],
    }
}

/// The page a `[[row]]` entry for a row whose list card is `vendor`'s cites.
fn row_source(vendor: &str) -> Option<&'static str> {
    Some(match vendor {
        "anthropic" => sources::ANTHROPIC_PRICING,
        "openai" => sources::OPENAI_PRICING,
        "xai" => sources::XAI_PRICING,
        "deepseek" => sources::DEEPSEEK_PRICING,
        "together" => sources::TOGETHER_PRICING,
        "groq" => sources::GROQ_MODELS,
        "fireworks" => sources::FIREWORKS_PRICING,
        _ => return None,
    })
}

fn successor(
    inputs: &Inputs,
    hosts: &Hosts,
    cat: &Catalog<'_>,
    g: &NewGroup,
) -> Result<Option<Finding>> {
    let pred = &cat.rows[g.pred];
    let v = &g.version;
    let vs = lineage::format_version(v, '.');
    let mut reasons: Vec<String> = Vec::new();
    let mut notes: Vec<String> = g
        .seen
        .iter()
        .map(|(h, id)| format!("{h} lists {id}"))
        .collect();
    notes.dedup();

    // Each predecessor host's id for the new version, as that host lists it.
    let mut kept: Vec<(bool, NewCand)> = Vec::new();
    let mut ids: BTreeMap<&'static str, String> = BTreeMap::new();
    for (i, (resp, c)) in arms(pred).into_iter().enumerate() {
        let h = host(c);
        if let Some(rule) = hand_rule(c) {
            reasons.push(format!(
                "the predecessor's {h} candidate {} has a hand-kept rule: {rule}",
                c.upstream_model
            ));
        }
        let Some(list) = hosts.lists.get(h) else {
            reasons.push(format!(
                "cannot tell whether {h} serves it: no {h} listing was read"
            ));
            continue;
        };
        match derive(list, c.upstream_model, v) {
            Derived::One(id) => {
                ids.insert(h, id.clone());
                kept.push((
                    resp,
                    NewCand {
                        provider: c.provider,
                        id,
                        path: c.path,
                    },
                ));
            }
            Derived::NotListed if i == 0 => {
                reasons.push(format!("the primary host {h} does not list version {vs}"))
            }
            Derived::NotListed => {
                notes.push(format!("{h} does not list it: that candidate is left out"))
            }
            Derived::Ambiguous(all) => reasons.push(format!(
                "{h} lists several ids for version {vs}: {}",
                all.join(", ")
            )),
            Derived::NoLineage => reasons.push(format!(
                "cannot derive the {h} id from {}",
                c.upstream_model
            )),
        }
    }
    let primary = ids.get(host(&pred.candidates[0])).cloned();
    let or_slug = ids.get("openrouter").cloned();

    // The row's name, spelled as the predecessor's is.
    let name = if pred.model == pred.candidates[0].upstream_model {
        primary.clone()
    } else if arms(pred)
        .iter()
        .any(|(_, c)| c.provider == ProviderId::OpenRouter && c.upstream_model == pred.model)
    {
        or_slug.clone()
    } else {
        None
    };
    let Some(name) = name.or_else(|| with_version(pred.model, v)) else {
        return Ok(None);
    };
    if cat.exists(&name) {
        return Ok(None);
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._/-".contains(&b))
    {
        reasons.push(format!("the derived row name {name:?} is not log-safe"));
    }
    if ids.iter().any(|(h, id)| cat.recorded(h, id)) {
        return Ok(None);
    }

    // A host the predecessor doesn't use that lists the new model.
    let pred_hosts: BTreeSet<&str> = arms(pred).iter().map(|(_, c)| host(c)).collect();
    let wanted: BTreeSet<String> = ids.values().map(|id| norm(id)).collect();
    for (&h, list) in &hosts.lists {
        if pred_hosts.contains(h) || h == "openrouter" || h == "bedrock" {
            continue;
        }
        if let Some(m) = list.iter().find(|m| wanted.contains(&norm(&m.id))) {
            reasons.push(format!(
                "new host: {h} lists it ({}), and {} has no {h} candidate",
                m.id, pred.model
            ));
        }
    }

    // Feature class, as OpenRouter describes both.
    let pred_or = arms(pred)
        .iter()
        .find(|(_, c)| c.provider == ProviderId::OpenRouter)
        .map(|(_, c)| c.upstream_model);
    match (
        pred_or.and_then(|s| hosts.or_cards.get(s)),
        or_slug.as_ref().and_then(|s| hosts.or_cards.get(s)),
    ) {
        (Some(a), Some(b)) => {
            let show = |v: &[String]| {
                if v.is_empty() {
                    "none".to_owned()
                } else {
                    v.join(", ")
                }
            };
            let num = |n: Option<u64>| n.map_or_else(|| "unlisted".to_owned(), |n| n.to_string());
            if a.input != b.input {
                reasons.push(format!(
                    "feature class differs: input modalities {} → {}",
                    show(&a.input),
                    show(&b.input)
                ));
            }
            if a.params != b.params {
                reasons.push(format!(
                    "feature class differs: supported parameters {} → {}",
                    show(&a.params),
                    show(&b.params)
                ));
            }
            if a.context != b.context {
                reasons.push(format!(
                    "feature class differs: context window {} → {}",
                    num(a.context),
                    num(b.context)
                ));
            }
            if a.max_output != b.max_output {
                reasons.push(format!(
                    "feature class differs: max output {} → {}",
                    num(a.max_output),
                    num(b.max_output)
                ));
            }
        }
        _ => reasons.push(
            "feature class unknown: OpenRouter lists the predecessor or the new model not at all"
                .into(),
        ),
    }

    // The truth file's rules for the predecessor.
    if let Some(row) = cat.truth_row(pred.model) {
        for k in ["input_limit", "output_limit"] {
            if row.get(k).is_some() {
                reasons.push(format!(
                    "the predecessor's card carries a measured {k}; the new model's is unmeasured"
                ));
            }
        }
        if tfield(row, "follows").is_some() {
            reasons.push("the predecessor's list price follows its maker's own card".into());
        }
    }
    let mut named: Vec<&str> = arms(pred).iter().map(|(_, c)| c.upstream_model).collect();
    if let Some(p) = cat.pricing(pred.model) {
        named.extend(p.cost.values().map(String::as_str));
        named.push(&p.list_card);
    }
    for m in cat.measured(&named) {
        reasons.push(format!("the predecessor has a recorded measurement in {m}"));
    }

    // Prices: each kept host's card, built from the snapshots as the generator would.
    let new_slugs: Vec<String> = or_slug.iter().cloned().collect();
    let store = match with_endpoints(inputs, cat, &new_slugs) {
        Ok(s) => Some(s),
        Err(e) => {
            reasons.push(format!("no rate: {e}"));
            None
        }
    };
    let Some(pricing) = cat.pricing(pred.model) else {
        return Err(format!("{}: no [[pricing]]", pred.model));
    };
    let display = with_version(pred.card.name, v);
    let d = Derive {
        cat,
        store: store.as_ref(),
        pred,
        v,
        name: &name,
        display: display.as_deref(),
    };
    let mut cost: Vec<(String, String)> = Vec::new();
    let mut cards: Vec<(String, String)> = Vec::new();
    let mut refresh: BTreeSet<String> = BTreeSet::new();
    // Host → its new card.
    let mut built: BTreeMap<String, Card> = BTreeMap::new();
    let mut add_card = |old: &str, spec: &CardSpec, refresh: &mut BTreeSet<String>| {
        if !cards
            .iter()
            .any(|(_, b)| b.contains(&format!("name = \"{}\"\n", spec.name)))
        {
            cards.push((old.to_owned(), catalog_edit::card_block(spec)));
        }
        for s in card_sources(spec.name.split(':').next().unwrap_or("")) {
            refresh.insert((*s).to_owned());
        }
    };
    for (_, nc) in &kept {
        let h = by_id(nc.provider).name;
        if cost.iter().any(|(x, _)| x == h) {
            continue;
        }
        let Some(want) = pricing.cost.get(h) else {
            reasons.push(format!("{}: [[pricing]] has no cost for {h}", pred.model));
            continue;
        };
        if want == "list" {
            cost.push((h.to_owned(), "list".into()));
        } else if let Some(r) = want.strip_prefix("unverified: ") {
            reasons.push(format!("{h}'s cost is unverified ({r})"));
        } else if let Some(old_slug) = want.strip_prefix("openrouter:") {
            let Some(s) = store.as_ref() else { continue };
            match (
                build::openrouter(s, cat.spec, old_slug),
                build::openrouter(s, cat.spec, &nc.id),
            ) {
                (Ok(a), Ok(b)) if !b.is_empty() => {
                    if or_tiers(&a) != or_tiers(&b) {
                        reasons.push(format!(
                            "feature class differs: OpenRouter {} {} → {}",
                            nc.id,
                            or_tiers(&a),
                            or_tiers(&b)
                        ));
                    }
                    cost.push((h.to_owned(), format!("openrouter:{}", nc.id)));
                }
                (Ok(_), Ok(_)) => reasons.push(format!(
                    "no rate: OpenRouter lists no endpoint for {}",
                    nc.id
                )),
                (Err(e), _) => reasons.push(format!("no rate for the predecessor: {e}")),
                (_, Err(e)) => reasons.push(format!("no rate: {e}")),
            }
        } else {
            match d.card(want, nc) {
                Ok((spec, card)) => {
                    add_card(want, &spec, &mut refresh);
                    cost.push((h.to_owned(), spec.name.clone()));
                    built.insert(h.to_owned(), card);
                }
                Err(e) => reasons.push(e),
            }
        }
    }
    // The list price: the list card's standard tier, or (an inline list price) the primary's
    // card's, where the predecessor's list price is exactly its primary's card.
    let p0 = host(&pred.candidates[0]);
    let mut list_name = String::from("list");
    let mut list_card: Option<Card> = None;
    if pricing.list_card == "list" {
        let old = pricing
            .cost
            .get(p0)
            .and_then(|n| cat.card_spec(n))
            .zip(store.as_ref())
            .and_then(|(cs, s)| build::card(s, cat.spec, cs).ok())
            .and_then(|c| list_price(&c).ok());
        if old.as_ref() == Some(&price_of(&pred.price)) {
            list_card = built.get(p0).cloned();
        } else {
            reasons.push(
                "the predecessor's list price is its maker's published rate, not a host card's"
                    .into(),
            );
        }
    } else {
        let list_host = pricing
            .cost
            .iter()
            .find(|(_, c)| c.as_str() == "list")
            .map(|(h, _)| h.as_str());
        match list_host.and_then(|h| kept.iter().find(|(_, c)| by_id(c.provider).name == h)) {
            Some((_, nc)) => match d.card(&pricing.list_card, nc) {
                Ok((spec, card)) => {
                    add_card(&pricing.list_card, &spec, &mut refresh);
                    list_name = spec.name.clone();
                    list_card = Some(card);
                }
                Err(e) => reasons.push(e),
            },
            None => reasons.push(format!(
                "the host of its list card ({}) does not list it",
                pricing.list_card
            )),
        }
    }
    let new_list_name = list_name;
    reasons.sort();
    reasons.dedup();

    // The release time, from the primary vendor's own listing where it has one.
    let created = match host(&pred.candidates[0]) {
        h @ ("anthropic" | "openai" | "xai") => primary
            .as_deref()
            .and_then(|id| hosts.find(h, id))
            .and_then(|m| m.created),
        _ => or_slug
            .as_deref()
            .and_then(|s| hosts.find("openrouter", s))
            .and_then(|m| m.created),
    };
    if created.is_none() {
        reasons.push("no release time in the sources".into());
    }
    let Some(display) = display else {
        reasons.push(format!(
            "cannot derive the display name from {:?}",
            pred.card.name
        ));
        let p = or_slug.as_deref().and_then(|s| hosts.or_price(s));
        return Ok(Some(human_successor(pred, &name, p, notes, reasons, hosts)));
    };

    let prices = list_card.as_ref().map(list_price).transpose()?;
    if prices.is_none() && !reasons.iter().any(|r| r.starts_with("no rate")) {
        reasons.push("no rate for the list card".into());
    }
    if !reasons.is_empty() {
        let p = prices.or_else(|| or_slug.as_deref().and_then(|s| hosts.or_price(s)));
        return Ok(Some(human_successor(pred, &name, p, notes, reasons, hosts)));
    }
    let (Some(prices), Some(created)) = (prices, created) else {
        return Ok(None);
    };

    // The pieces of the edit.
    let cands: Vec<NewCand> = kept
        .iter()
        .filter(|(r, _)| !r)
        .map(|(_, c)| c.clone())
        .collect();
    let resps: Vec<NewCand> = kept
        .iter()
        .filter(|(r, _)| *r)
        .map(|(_, c)| c.clone())
        .collect();
    let after = cat
        .rows
        .iter()
        .map(|r| r.model)
        .filter(|m| *m < name.as_str())
        .max()
        .map(str::to_owned);
    let row = catalog_edit::row(&catalog_edit::NewRow {
        model: &name,
        wire: pred.wire,
        candidates: &cands,
        responses: &resps,
        price: &prices,
        card: pred.card,
        name: &display,
        created,
    });
    let truth_row = cat.truth_row(pred.model).map(|r| {
        let vendor = new_list_name.split(':').next().unwrap_or("").to_owned();
        let vendor = if new_list_name == "list" {
            host(&pred.candidates[0]).to_owned()
        } else {
            vendor
        };
        (
            pred.model.to_owned(),
            catalog_edit::truth_row(
                &name,
                row_source(&vendor).unwrap_or(sources::OPENROUTER_PRICING),
                &inputs.today,
                &prices,
                created,
                r,
            ),
        )
    });
    let pricing_block = catalog_edit::pricing_block(&name, &new_list_name, &cost);
    let mut providers: Vec<String> = cost.iter().map(|(h, _)| h.clone()).collect();
    providers.sort();
    let mut sources_list: Vec<String> = g
        .seen
        .iter()
        .filter_map(|(h, _)| hosts.urls.get(h).map(|u| (*u).to_owned()))
        .collect();
    sources_list.sort();
    sources_list.dedup();
    let mut refresh: Vec<String> = refresh.into_iter().collect();
    refresh.push("openrouter.models".into());
    for s in &new_slugs {
        refresh.push(format!("openrouter.endpoints/{s}"));
    }
    notes.push(format!(
        "generated from {}: {} candidate(s), list ${} / ${} per MTok",
        pred.model,
        cands.len(),
        prices[0],
        prices[1]
    ));
    Ok(Some(Finding {
        slug: slug("add-", &name),
        class: Class::Successor,
        model: name.clone(),
        vendor: pred.card.owned_by.to_owned(),
        title: format!("catalog: add {name} (succeeds {})", pred.model),
        prices: Some(prices),
        sources: sources_list,
        notes,
        reasons: Vec::new(),
        edit: Some(Edit::Add(Addition {
            model: name,
            after,
            row,
            truth_row,
            cards,
            pricing: pricing_block,
            refresh,
            new_slugs,
        })),
        providers,
    }))
}

/// Re-spelling the predecessor's cards for the new version.
struct Derive<'a> {
    cat: &'a Catalog<'a>,
    store: Option<&'a Store>,
    pred: &'a ModelRoute,
    v: &'a [u32],
    name: &'a str,
    display: Option<&'a str>,
}

impl Derive<'_> {
    /// The new model's card on `nc`'s host, from the predecessor's card `old`: its spec (name and
    /// vendor row key re-spelled for the new version) and the card the snapshots build. An error
    /// is a needs-human reason: no rate, or a different set of pricing tiers.
    fn card(&self, old: &str, nc: &NewCand) -> std::result::Result<(CardSpec, Card), String> {
        let cs = self
            .cat
            .card_spec(old)
            .ok_or_else(|| format!("no rate: no [[card]] {old}"))?;
        let pred_id = arms(self.pred)
            .into_iter()
            .find(|(_, c)| c.provider == nc.provider)
            .map(|(_, c)| c.upstream_model)
            .unwrap_or_default();
        let respell = |s: &str| -> Option<String> {
            if s == pred_id {
                Some(nc.id.clone())
            } else if s == self.pred.model {
                Some(self.name.to_owned())
            } else if s == self.pred.card.name {
                self.display.map(str::to_owned)
            } else {
                with_version(s, self.v)
            }
        };
        let (vendor, key) = old
            .split_once(':')
            .ok_or_else(|| format!("no rate: [[card]] {old} is not vendor:model"))?;
        let name = respell(key)
            .map(|k| format!("{vendor}:{k}"))
            .ok_or_else(|| format!("cannot derive the card name from {old}"))?;
        let row = respell(&cs.row)
            .ok_or_else(|| format!("cannot derive {old}'s vendor row from {:?}", cs.row))?;
        let spec = CardSpec {
            name,
            row,
            ..cs.clone()
        };
        let store = self.store.ok_or("no rate: the snapshots did not load")?;
        let before =
            build::card(store, self.cat.spec, cs).map_err(|e| format!("no rate: {old}: {e}"))?;
        let after = build::card(store, self.cat.spec, &spec)
            .map_err(|e| format!("no rate: {}: {e}", spec.name))?;
        if tiers(&before) != tiers(&after) {
            return Err(format!(
                "feature class differs: {} pricing tiers {} → {}",
                spec.name,
                tiers(&before),
                tiers(&after)
            ));
        }
        Ok((spec, after))
    }
}

/// A new model no rule can add: its list price, or OpenRouter's (input and output only).
fn human_successor(
    pred: &ModelRoute,
    name: &str,
    prices: Option<[String; 4]>,
    notes: Vec<String>,
    reasons: Vec<String>,
    hosts: &Hosts,
) -> Finding {
    let mut sources: Vec<String> = notes
        .iter()
        .filter_map(|n| n.split(' ').next())
        .filter_map(|h| hosts.urls.get(h).map(|u| (*u).to_owned()))
        .collect();
    sources.sort();
    sources.dedup();
    Finding {
        slug: slug("add-", name),
        class: Class::NeedsHuman,
        model: name.to_owned(),
        vendor: pred.card.owned_by.to_owned(),
        title: format!("{name} (would succeed {})", pred.model),
        prices,
        sources,
        notes,
        reasons,
        edit: None,
        providers: Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------
// Retirements
// ---------------------------------------------------------------------------------------------

/// One (date, source, why) a candidate retires on.
#[derive(Clone, Debug)]
struct Due {
    date: String,
    source: String,
    why: String,
}

/// `October 23, 2026`, `Oct 1, 2026`, `2026-09-24` → `YYYY-MM-DD`; anything else (`Not sooner
/// than …`, `To be announced`) is not a date.
fn doc_date(cell: &str) -> Option<String> {
    let s = cell.trim().replace('\u{2011}', "-");
    if s.len() == 10 && build::day(&s).is_ok() {
        return Some(s);
    }
    const MONTHS: [&str; 12] = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    let mut words = s.split([' ', ',']).filter(|w| !w.is_empty());
    let (m, d, y) = (words.next()?, words.next()?, words.next()?);
    if words.next().is_some() {
        return None;
    }
    let m = MONTHS
        .iter()
        .position(|p| m.len() >= 3 && m.to_ascii_lowercase().starts_with(p))?
        + 1;
    let (d, y): (u32, u32) = (d.parse().ok()?, y.parse().ok()?);
    Some(format!("{y:04}-{m:02}-{d:02}"))
}

/// Every (model id, retirement date) a deprecation page's tables list: a date column (the one
/// that says retirement, shutdown or removal, if any says so) and a model column (not the
/// replacement), outside fine-tuning sections. A cell may hold several backticked ids.
pub fn deprecations(md: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Ok(tables) = crate::md::tables(md) else {
        return out;
    };
    for t in tables {
        if t.context
            .iter()
            .any(|c| c.starts_with('#') && c.to_ascii_lowercase().contains("fine-tun"))
        {
            continue;
        }
        let lower: Vec<String> = t.header.iter().map(|c| c.to_ascii_lowercase()).collect();
        let date_col = lower
            .iter()
            .position(|c| {
                c.contains("date")
                    && ["retire", "shutdown", "removal"]
                        .iter()
                        .any(|w| c.contains(w))
            })
            .or_else(|| lower.iter().position(|c| c.contains("date")));
        let model_col = lower.iter().position(|c| {
            (c.contains("model") || c == "system")
                && !["replacement", "price", "type"]
                    .iter()
                    .any(|w| c.contains(w))
        });
        let (Some(dc), Some(mc)) = (date_col, model_col) else {
            continue;
        };
        for r in &t.rows {
            let Some(date) = doc_date(&r[dc]) else {
                continue;
            };
            let cell = &r[mc];
            let ids: Vec<&str> = if cell.contains('`') {
                cell.split('`').skip(1).step_by(2).collect()
            } else {
                vec![cell.as_str()]
            };
            for id in ids
                .into_iter()
                .map(str::trim)
                .filter(|i| !i.is_empty() && !i.contains(' '))
            {
                out.push((id.to_owned(), date.clone()));
            }
        }
    }
    out
}

fn retirements(inputs: &Inputs, hosts: &Hosts, cat: &Catalog<'_>) -> Result<Vec<Finding>> {
    let horizon = add_days(&inputs.today, RETIRE_AHEAD_DAYS)?;
    // (row, arm index) → its dues.
    let mut dues: BTreeMap<(usize, usize), Vec<Due>> = BTreeMap::new();
    let mut vanished: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    let pages: BTreeMap<&str, Vec<(String, String)>> = DEPRECATION_PAGES
        .iter()
        .filter_map(|(p, _)| inputs.deprecations.get(*p).map(|md| (*p, deprecations(md))))
        .collect();
    for (ri, row) in cat.rows.iter().enumerate() {
        for (ai, (_, c)) in arms(row).into_iter().enumerate() {
            let h = host(c);
            let id = c.upstream_model;
            let mut add = |d: Due| dues.entry((ri, ai)).or_default().push(d);
            if hosts.lists.contains_key(h) {
                match hosts.find(h, id) {
                    Some(m) => {
                        if let Some(date) = &m.retires {
                            add(Due {
                                date: date.clone(),
                                source: hosts.urls.get(h).copied().unwrap_or_default().to_owned(),
                                why: format!("{h}'s listing retires {} on {date}", m.id),
                            });
                        }
                    }
                    None => vanished.entry(h).or_default().push((ri, ai)),
                }
            }
            if let Some(list) = pages.get(h) {
                for (listed, date) in list {
                    // An Anthropic alias pins one dated snapshot, so the snapshot's retirement is
                    // the alias's; an OpenAI alias moves between snapshots, so only its own id
                    // retires it.
                    let pinned = h == "anthropic" && undated(listed).eq_ignore_ascii_case(id);
                    if listed.eq_ignore_ascii_case(id) || pinned {
                        let url = DEPRECATION_PAGES
                            .iter()
                            .find(|(p, _)| *p == h)
                            .map_or("", |(_, u)| *u);
                        add(Due {
                            date: date.clone(),
                            source: url.trim_end_matches(".md").to_owned(),
                            why: format!("{h}'s deprecation page retires {listed} on {date}"),
                        });
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for (h, gone) in &vanished {
        // A candidate listed under both arms of a row counts once.
        let distinct: BTreeSet<(&str, &str)> = gone
            .iter()
            .map(|(ri, ai)| {
                (
                    cat.rows[*ri].model,
                    arms(&cat.rows[*ri])[*ai].1.upstream_model,
                )
            })
            .collect();
        if distinct.len() > VANISH_LIMIT {
            let list: Vec<String> = distinct
                .iter()
                .map(|(r, id)| format!("{id} ({r})"))
                .collect();
            out.push(Finding {
                slug: slug("vanished-", h),
                class: Class::NeedsHuman,
                model: format!("{} {h} candidates", distinct.len()),
                vendor: (*h).to_owned(),
                title: format!("{} candidates missing from {h}'s listing", distinct.len()),
                prices: None,
                sources: vec![hosts.urls.get(h).copied().unwrap_or_default().to_owned()],
                notes: list,
                reasons: vec![format!(
                    "{} candidates vanished from {h}'s listing at once (more than {VANISH_LIMIT}): a broken listing is likelier than {} retirements",
                    distinct.len(),
                    distinct.len()
                )],
                edit: None,
                providers: Vec::new(),
            });
            continue;
        }
        for &(ri, ai) in gone {
            dues.entry((ri, ai)).or_default().push(Due {
                date: inputs.today.clone(),
                source: hosts.urls.get(h).copied().unwrap_or_default().to_owned(),
                why: format!(
                    "{h}'s listing no longer has {}",
                    arms(&cat.rows[ri])[ai].1.upstream_model
                ),
            });
        }
    }
    // Per row: the arms due by the horizon, at their earliest date.
    let mut by_row: BTreeMap<usize, BTreeMap<usize, Due>> = BTreeMap::new();
    for ((ri, ai), ds) in dues {
        if let Some(d) = ds
            .into_iter()
            .filter(|d| d.date <= horizon)
            .min_by(|a, b| a.date.cmp(&b.date))
        {
            by_row.entry(ri).or_default().insert(ai, d);
        }
    }
    for (ri, due) in by_row {
        out.push(retire_row(inputs, cat, ri, &due)?);
    }
    Ok(out)
}

fn retire_row(
    inputs: &Inputs,
    cat: &Catalog<'_>,
    ri: usize,
    due: &BTreeMap<usize, Due>,
) -> Result<Finding> {
    let row = &cat.rows[ri];
    let all = arms(row);
    // A provider listed under both arms (a GPT row's OpenAI) retires as one.
    let mut gone: Vec<(ProviderId, &str)> = Vec::new();
    for &ai in due.keys() {
        let k = (all[ai].1.provider, all[ai].1.upstream_model);
        if !gone.contains(&k) {
            gone.push(k);
        }
    }
    let retiring = |c: &catalog::Candidate| gone.contains(&(c.provider, c.upstream_model));
    let date = due
        .values()
        .map(|d| d.date.as_str())
        .min()
        .unwrap_or_default()
        .to_owned();
    let mut notes: Vec<String> = due.values().map(|d| d.why.clone()).collect();
    notes.sort();
    notes.dedup();
    let mut sources: Vec<String> = due.values().map(|d| d.source.clone()).collect();
    sources.sort();
    sources.dedup();
    let what: Vec<String> = gone
        .iter()
        .map(|(p, id)| format!("{}/{id}", by_id(*p).name))
        .collect();
    let mut reasons = Vec::new();
    let whole = all.iter().all(|(_, c)| retiring(c));
    let mut successor = None;
    if whole {
        let line = row_line(row.model);
        successor = line.and_then(|(l, v)| {
            cat.rows
                .iter()
                .filter_map(|r| {
                    row_line(r.model)
                        .filter(|(x, w)| *x == l && *w > v)
                        .map(|(_, w)| (w, r.model))
                })
                .min()
                .map(|(_, m)| m)
        });
        if successor.is_none() {
            reasons.push(format!(
                "every candidate of {} retires and no newer row in its line exists to alias it to",
                row.model
            ));
        }
    } else if retiring(&row.candidates[0]) {
        reasons.push(format!(
            "the primary {}/{} retires; the row's wire, card and list price follow its primary",
            host(&row.candidates[0]),
            row.candidates[0].upstream_model
        ));
    }
    for (_, c) in &all {
        if retiring(c)
            && !whole
            && let Some(rule) = hand_rule(c)
        {
            reasons.push(format!(
                "{}/{} has a hand-kept rule: {rule}",
                host(c),
                c.upstream_model
            ));
        }
    }
    let pricing = cat
        .pricing(row.model)
        .ok_or_else(|| format!("{}: no [[pricing]]", row.model))?;
    if !whole {
        let mut needles: Vec<String> = Vec::new();
        for (p, id) in &gone {
            needles.push((*id).to_owned());
            if let Some(c) = pricing.cost.get(by_id(*p).name) {
                needles.push(c.clone());
            }
        }
        let n: Vec<&str> = needles.iter().map(String::as_str).collect();
        for m in cat.measured(&n) {
            reasons.push(format!(
                "a retiring candidate has a recorded measurement in {m}"
            ));
        }
    }
    if let Some(s) = successor
        && arms(cat.rows.iter().find(|r| r.model == s).unwrap_or(row))
            .iter()
            .all(|(_, c)| gone.contains(&(c.provider, c.upstream_model)))
    {
        reasons.push(format!("the successor {s} retires too"));
    }
    reasons.sort();
    reasons.dedup();
    let title_what = if whole {
        format!("retire {} ({date})", row.model)
    } else {
        format!("retire {} from {} ({date})", what.join(", "), row.model)
    };
    if !reasons.is_empty() {
        return Ok(Finding {
            slug: slug("retire-", row.model),
            class: Class::NeedsHuman,
            model: row.model.to_owned(),
            vendor: row.card.owned_by.to_owned(),
            title: title_what,
            prices: Some(price_of(&row.price)),
            sources,
            notes,
            reasons,
            edit: None,
            providers: Vec::new(),
        });
    }

    // The [[retired]] entries: one per retiring (provider, id) not already recorded; on a whole
    // row the first also names the row.
    let mut retired = Vec::new();
    for (i, (p, id)) in gone.iter().enumerate() {
        let h = by_id(*p).name;
        if cat.retired_entry(h, id) && !whole {
            continue;
        }
        let d = due
            .iter()
            .find(|(ai, _)| all[**ai].1.provider == *p && all[**ai].1.upstream_model == *id)
            .map(|(_, d)| d);
        let Some(d) = d else { continue };
        let note = if whole {
            format!(
                "{}; the row was removed by catalog-drift and {} is now an alias of {}.",
                d.why,
                row.model,
                successor.unwrap_or_default()
            )
        } else {
            format!(
                "{}; removed from the {} row by catalog-drift.",
                d.why, row.model
            )
        };
        retired.push(catalog_edit::retired_block(
            (whole && i == 0).then_some(row.model),
            h,
            id,
            &d.source,
            &d.date,
            &inputs.today,
            &note,
        ));
    }
    let keep = |resp: bool| -> Vec<NewCand> {
        all.iter()
            .filter(|(r, c)| *r == resp && !retiring(c))
            .map(|(_, c)| NewCand {
                provider: c.provider,
                id: c.upstream_model.to_owned(),
                path: c.path,
            })
            .collect()
    };
    let (cands, resps) = (keep(false), keep(true));
    let mut providers: Vec<String> = cands
        .iter()
        .chain(&resps)
        .map(|c| by_id(c.provider).name.to_owned())
        .collect();
    providers.sort();
    providers.dedup();
    // Cards no row uses once this one's costs are gone.
    let mut cost: Vec<(String, String)> = Vec::new();
    for (h, c) in &pricing.cost {
        if whole || !providers.contains(h) {
            continue;
        }
        cost.push((h.clone(), c.clone()));
    }
    let left: BTreeSet<&str> = cost.iter().map(|(_, c)| c.as_str()).collect();
    let mut drop_cards: Vec<String> = pricing
        .cost
        .values()
        .chain(std::iter::once(&pricing.list_card))
        .filter(|c| cat.card_spec(c).is_some())
        .filter(|c| whole || (!left.contains(c.as_str()) && **c != pricing.list_card))
        .filter(|c| {
            !cat.spec.pricing.iter().any(|p| {
                p.model != row.model && (p.list_card == **c || p.cost.values().any(|x| x == *c))
            })
        })
        .cloned()
        .collect();
    drop_cards.sort();
    drop_cards.dedup();
    Ok(Finding {
        slug: slug("retire-", row.model),
        class: Class::Retirement,
        model: row.model.to_owned(),
        vendor: row.card.owned_by.to_owned(),
        title: format!("catalog: {title_what}"),
        prices: Some(price_of(&row.price)),
        sources,
        notes,
        reasons: Vec::new(),
        edit: Some(Edit::Retire(Removal {
            model: row.model.to_owned(),
            candidates: (!whole).then(|| catalog_edit::candidates(&cands)),
            responses: (!whole && !row.responses.is_empty() && resps.len() != row.responses.len())
                .then(|| catalog_edit::candidates(&resps)),
            alias_to: if whole {
                successor.map(str::to_owned)
            } else {
                None
            },
            cost: (!whole).then(|| catalog_edit::cost_line(&cost)),
            drop_cards,
            retired,
        })),
        providers: if whole {
            successor
                .and_then(|s| cat.rows.iter().find(|r| r.model == s))
                .map(|r| {
                    let mut p: Vec<String> =
                        arms(r).iter().map(|(_, c)| host(c).to_owned()).collect();
                    p.sort();
                    p.dedup();
                    p
                })
                .unwrap_or_default()
        } else {
            providers
        },
    })
}

// ---------------------------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------------------------

fn md_cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn prices_text(p: Option<&[String; 4]>) -> String {
    match p {
        None => "unknown".into(),
        Some([i, o, r, w]) if r.is_empty() && w.is_empty() => format!("${i} in / ${o} out"),
        Some([i, o, r, w]) => format!("${i} in / ${o} out / ${r} cache read / ${w} cache write"),
    }
}

impl Finding {
    /// The PR body (or the issue entry) for this finding, before the live result.
    pub fn markdown(&self) -> String {
        let mut s = format!(
            "**{}** ({}, {})\n\n",
            self.title,
            self.class.as_str(),
            self.vendor
        );
        s.push_str(&format!(
            "- List price (USD per MTok): {}\n",
            prices_text(self.prices.as_ref())
        ));
        for n in &self.notes {
            s.push_str(&format!("- {n}\n"));
        }
        for r in &self.reasons {
            s.push_str(&format!("- needs a human: {r}\n"));
        }
        for u in &self.sources {
            s.push_str(&format!("- source: {u}\n"));
        }
        s
    }

    pub fn to_json(&self) -> Value {
        json!({
            "slug": self.slug,
            "class": self.class.as_str(),
            "model": self.model,
            "vendor": self.vendor,
            "title": self.title,
            "prices": self.prices.as_ref().map(|[i, o, r, w]| json!({
                "input": i, "output": o, "cache_read": r, "cache_write": w
            })),
            "sources": self.sources,
            "notes": self.notes,
            "reasons": self.reasons,
            "providers": self.providers,
            "markdown": self.markdown(),
        })
    }
}

/// The body of the one open `catalog-drift` issue: every needs-human finding.
pub fn issue_markdown(findings: &[Finding]) -> String {
    let humans: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.class == Class::NeedsHuman)
        .collect();
    let mut s = String::from(
        "Catalog drift a rule cannot settle (`rates-sync catalog-drift`, \
         `.github/workflows/catalog-drift.yml`). Fix each by hand (add the row, record it in \
         `verify/catalog_truth.toml` `[[not_carried]]` or `[[retired]]`, or edit the row), and \
         the next run drops it.\n\n",
    );
    if humans.is_empty() {
        s.push_str("Nothing needs a human.\n");
        return s;
    }
    s.push_str("| Finding | Vendor | List price (USD/MTok) | Source | Why it needs a human |\n");
    s.push_str("| ------- | ------ | --------------------- | ------ | -------------------- |\n");
    for f in humans {
        s.push_str(&format!(
            "| `{}`: {} | {} | {} | {} | {} |\n",
            f.slug,
            md_cell(&f.title),
            md_cell(&f.vendor),
            md_cell(&prices_text(f.prices.as_ref())),
            md_cell(&f.sources.join(" ")),
            md_cell(&f.reasons.join("; "))
        ));
    }
    s
}

// ---------------------------------------------------------------------------------------------
// Saving and loading inputs
// ---------------------------------------------------------------------------------------------

const SOURCES_DIR: &str = "sources";

impl Inputs {
    /// Read a directory written by [`Inputs::save`] (the store under `sources/`).
    pub fn load(dir: &Path, defs: &[sources::SourceDef]) -> Result<Inputs> {
        let read = |p: &str| std::fs::read_to_string(dir.join(p)).ok();
        let need = |p: &str| read(p).ok_or_else(|| format!("{}: missing", dir.join(p).display()));
        let mut endpoints = BTreeMap::new();
        let edir = dir.join("endpoints");
        walk(&edir, &edir, &mut |rel, text| {
            if let Some(slug) = rel.strip_suffix(".json") {
                endpoints.insert(slug.to_owned(), text);
            }
        })?;
        let mut deprecations = BTreeMap::new();
        for (p, _) in DEPRECATION_PAGES {
            if let Some(t) = read(&format!("deprecations/{p}.md")) {
                deprecations.insert((*p).to_owned(), t);
            }
        }
        Ok(Inputs {
            today: need("today")?.trim().to_owned(),
            store: crate::snapshot::load(&dir.join(SOURCES_DIR), defs)?,
            openrouter: need("openrouter-models.json")?,
            endpoints,
            anthropic: read("anthropic-models.json"),
            openai: read("openai-models.json"),
            bedrock: read("bedrock-profiles.json"),
            deprecations,
        })
    }
}

fn walk(root: &Path, dir: &Path, f: &mut dyn FnMut(&str, String)) -> Result<()> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    let mut paths: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    paths.sort();
    for p in paths {
        if p.is_dir() {
            walk(root, &p, f)?;
        } else if let Ok(rel) = p.strip_prefix(root) {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            f(&rel.to_string_lossy().replace('\\', "/"), text);
        }
    }
    Ok(())
}

/// The sources directory of a saved inputs directory.
pub fn sources_dir(dir: &Path) -> std::path::PathBuf {
    dir.join(SOURCES_DIR)
}
