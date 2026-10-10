//! One strict reader per primary source. Each finds its table by its exact header (and label),
//! keeps the rows as text, and parses a row's cells only when a card asks for that row, so an
//! unrelated row (an audio model, a "Free" cell) never fails generation, while a changed header,
//! a missing row, a duplicate row or a malformed cell on a row we price always does.

use crate::Result;
use crate::dec::{Rate, float_per_million};
use crate::md::{self, Table, collapse, dollars, html_rows, strip_sup, tables};
use serde_json::Value;
use std::collections::BTreeMap;

/// One tier as a vendor publishes it. `None` is "not published" (a `-` cell, an absent field).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Published {
    pub input: Rate,
    pub output: Option<Rate>,
    pub cache_read: Option<Rate>,
    pub cache_write: Option<Rate>,
    pub cache_write_1h: Option<Rate>,
}

fn one<'a, T>(hits: Vec<&'a T>, what: &str) -> Result<&'a T> {
    match hits[..] {
        [x] => Ok(x),
        [] => Err(format!("{what}: not found")),
        _ => Err(format!("{what}: listed {} times", hits.len())),
    }
}

/// `-` / `—` is unpublished; anything else must be a dollar amount.
fn opt_dollars(cell: &str) -> Result<Option<Rate>> {
    match cell.trim() {
        "-" | "—" | "\\-" => Ok(None),
        c => dollars(c).map(Some),
    }
}

// --- Anthropic: platform.claude.com/docs/en/about-claude/pricing.md --------------------------

pub struct Anthropic {
    models: Table,
    fast: Table,
}

const ANTHROPIC_HEADER: &[&str] = &[
    "Model",
    "Base input tokens",
    "5m cache writes",
    "1h cache writes",
    "Cache hits and refreshes",
    "Output tokens",
];

pub fn anthropic(md: &str) -> Result<Anthropic> {
    let ts = tables(md)?;
    let models = one(
        ts.iter()
            .filter(|t| {
                t.header
                    .iter()
                    .map(String::as_str)
                    .eq(ANTHROPIC_HEADER.iter().copied())
            })
            .collect(),
        "Anthropic pricing: the model pricing table",
    )?
    .clone();
    let fast = one(
        ts.iter()
            .filter(|t| t.context.iter().any(|c| c == "### Fast mode pricing"))
            .collect(),
        "Anthropic pricing: the fast mode table",
    )?
    .clone();
    fast.expect_header(&["Model", "Input", "Output"], "Anthropic fast mode table")?;
    Ok(Anthropic { models, fast })
}

/// `$10 / MTok`, footnote markers allowed.
fn mtok(cell: &str) -> Result<Rate> {
    let c = strip_sup(cell);
    let n = c
        .trim()
        .strip_suffix(" / MTok")
        .ok_or_else(|| format!("{cell:?} is not `$x / MTok`"))?;
    dollars(n)
}

/// The model name a row is keyed by: `Claude Opus 4.1 ([retired, …](…))` is `Claude Opus 4.1`.
fn anthropic_name(cell: &str) -> &str {
    cell.split(" ([").next().unwrap_or(cell).trim()
}

impl Anthropic {
    /// The standard row: input, 5-minute write, 1-hour write, cache hit, output.
    pub fn model(&self, name: &str) -> Result<Published> {
        let r = one(
            self.models
                .rows
                .iter()
                .filter(|r| anthropic_name(&r[0]) == name)
                .collect(),
            &format!("Anthropic model pricing row {name:?}"),
        )?;
        Ok(Published {
            input: mtok(&r[1])?,
            cache_write: Some(mtok(&r[2])?),
            cache_write_1h: Some(mtok(&r[3])?),
            cache_read: Some(mtok(&r[4])?),
            output: Some(mtok(&r[5])?),
        })
    }

    /// The fast mode (input, output), if the model is in the fast mode table. A row may name
    /// several models (`Claude Opus 5 / Claude Opus 4.8`).
    pub fn fast(&self, name: &str) -> Result<Option<(Rate, Rate)>> {
        let hits: Vec<&Vec<String>> = self
            .fast
            .rows
            .iter()
            .filter(|r| r[0].split(" / ").any(|n| n.trim() == name))
            .collect();
        match hits[..] {
            [] => Ok(None),
            [r] => Ok(Some((mtok(&r[1])?, mtok(&r[2])?))),
            _ => Err(format!("Anthropic fast mode: {name:?} listed twice")),
        }
    }
}

// --- OpenAI: developers.openai.com/api/docs/pricing.md ---------------------------------------

/// One OpenAI tier for one model: the short-context rates and, where published, the long.
#[derive(Clone, Copy, Debug)]
pub struct OaTier {
    pub short: Published,
    pub long: Option<Published>,
}

pub struct OpenAi {
    /// `standard` / `flex` / `fast` / `ultrafast` → the flagship table.
    flagship: BTreeMap<&'static str, Table>,
    /// `standard` / `fast` → the specialized-models table.
    specialized: BTreeMap<&'static str, Table>,
}

const OA_FLAGSHIP: &[&str] = &[
    "Model",
    "Short context input",
    "Short context cached input",
    "Short context cache writes",
    "Short context output",
    "Long context input",
    "Long context cached input",
    "Long context cache writes",
    "Long context output",
];
const OA_SPECIALIZED: &[&str] = &["Category", "Model", "Input", "Cached input", "Output"];

pub fn openai(md: &str) -> Result<OpenAi> {
    let ts = tables(md)?;
    let mut flagship = BTreeMap::new();
    for (tier, label) in [
        ("standard", "### Standard pricing data"),
        ("flex", "### Flex pricing data"),
        ("fast", "### Fast pricing data"),
        ("ultrafast", "### Ultrafast pricing data"),
    ] {
        let t = md::table_labelled(&ts, label).map_err(|e| format!("OpenAI pricing: {e}"))?;
        t.expect_header(OA_FLAGSHIP, label)?;
        flagship.insert(tier, t.clone());
    }
    let mut specialized = BTreeMap::new();
    let spec: Vec<&Table> = ts
        .iter()
        .filter(|t| {
            t.header
                .iter()
                .map(String::as_str)
                .eq(OA_SPECIALIZED.iter().copied())
        })
        .collect();
    for t in spec {
        let tier = match t.context.get(1).map(String::as_str) {
            Some("Standard") => "standard",
            Some("Fast") => "fast",
            other => {
                return Err(format!(
                    "OpenAI pricing: a specialized-models table under {other:?} (line {})",
                    t.line
                ));
            }
        };
        if specialized.insert(tier, t.clone()).is_some() {
            return Err(format!("OpenAI pricing: two specialized {tier} tables"));
        }
    }
    if !specialized.contains_key("standard") {
        return Err("OpenAI pricing: no specialized-models Standard table".into());
    }
    Ok(OpenAi {
        flagship,
        specialized,
    })
}

/// `gpt-5.5 (<272K context length)` is `gpt-5.5`.
fn oa_name(cell: &str) -> &str {
    cell.strip_suffix(" (<272K context length)")
        .unwrap_or(cell)
        .trim()
}

impl OpenAi {
    /// The model's rates in one tier, from the flagship table or else the specialized one; `None`
    /// when the tier does not list the model.
    pub fn tier(&self, tier: &str, model: &str) -> Result<Option<OaTier>> {
        let what = format!("OpenAI {tier} pricing for {model:?}");
        let flag: Vec<&Vec<String>> = self
            .flagship
            .get(tier)
            .map(|t| t.rows.iter().filter(|r| oa_name(&r[0]) == model).collect())
            .unwrap_or_default();
        let spec: Vec<&Vec<String>> = self
            .specialized
            .get(tier)
            .map(|t| t.rows.iter().filter(|r| r[1] == model).collect())
            .unwrap_or_default();
        match (&flag[..], &spec[..]) {
            ([], []) => Ok(None),
            ([r], []) => {
                let short = Published {
                    input: dollars(&r[1])?,
                    cache_read: opt_dollars(&r[2])?,
                    cache_write: opt_dollars(&r[3])?,
                    output: Some(dollars(&r[4])?),
                    cache_write_1h: None,
                };
                let long_cells = [&r[5], &r[6], &r[7], &r[8]];
                let long = if long_cells.iter().all(|c| opt_dollars(c).ok() == Some(None)) {
                    None
                } else {
                    Some(Published {
                        input: dollars(&r[5])?,
                        cache_read: opt_dollars(&r[6])?,
                        cache_write: opt_dollars(&r[7])?,
                        output: Some(dollars(&r[8])?),
                        cache_write_1h: None,
                    })
                };
                Ok(Some(OaTier { short, long }))
            }
            ([], [r]) => Ok(Some(OaTier {
                short: Published {
                    input: dollars(&r[2])?,
                    cache_read: opt_dollars(&r[3])?,
                    output: opt_dollars(&r[4])?,
                    cache_write: None,
                    cache_write_1h: None,
                },
                long: None,
            })),
            _ => Err(format!("{what}: listed more than once")),
        }
    }
}

// --- xAI: GET /v1/language-models, cross-checked with docs.x.ai/developers/pricing.md ---------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XaiModel {
    pub short: Published,
    pub long: Published,
    pub threshold: u64,
}

/// The model the API lists under `name` (its id or one of its aliases).
pub fn xai_api(json: &str, name: &str) -> Result<(String, XaiModel)> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("xAI models: {e}"))?;
    let models = v
        .get("models")
        .and_then(Value::as_array)
        .ok_or("xAI models: no `models`")?;
    let hits: Vec<&Value> = models
        .iter()
        .filter(|m| {
            m.get("id").and_then(Value::as_str) == Some(name)
                || m.get("aliases")
                    .and_then(Value::as_array)
                    .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(name)))
        })
        .collect();
    let m = one(hits, &format!("xAI language model {name:?}"))?;
    let int = |k: &str| -> Result<u64> {
        m.get(k)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("xAI {name}: no integer `{k}`"))
    };
    let tier = |p: &str, c: &str, o: &str| -> Result<Published> {
        let input = Rate::xai_ticks(int(p)?);
        Ok(Published {
            input,
            cache_read: Some(Rate::xai_ticks(int(c)?)),
            output: Some(Rate::xai_ticks(int(o)?)),
            cache_write: None,
            cache_write_1h: None,
        })
    };
    let id = m
        .get("id")
        .and_then(Value::as_str)
        .ok_or("xAI: a model without an id")?
        .to_owned();
    Ok((
        id,
        XaiModel {
            short: tier(
                "prompt_text_token_price",
                "cached_prompt_text_token_price",
                "completion_text_token_price",
            )?,
            long: tier(
                "prompt_text_token_price_long_context",
                "cached_prompt_text_token_price_long_context",
                "completion_text_token_price_long_context",
            )?,
            threshold: int("long_context_threshold")?,
        },
    ))
}

/// The pricing page's two rows for `id`: (`< threshold`, `≥ threshold`), and the threshold the
/// rows name (`200k` is 200,000).
pub fn xai_page(md: &str, id: &str) -> Result<(Published, Published, u64)> {
    let ts = tables(md)?;
    let t =
        md::table_labelled(&ts, "### Text API Pricing").map_err(|e| format!("xAI pricing: {e}"))?;
    t.expect_header(
        &[
            "Model",
            "Context",
            "Input / 1M tokens",
            "Cached input / 1M tokens",
            "Output / 1M tokens",
        ],
        "xAI text pricing",
    )?;
    let mut below = Vec::new();
    let mut above = Vec::new();
    for r in &t.rows {
        let Some((name, rest)) = r[0].split_once(" (") else {
            return Err(format!(
                "xAI pricing: row {:?} has no prompt-size bracket",
                r[0]
            ));
        };
        if name != id {
            continue;
        }
        let p = Published {
            input: dollars(&r[2])?,
            cache_read: Some(dollars(&r[3])?),
            output: Some(dollars(&r[4])?),
            cache_write: None,
            cache_write_1h: None,
        };
        let k = |s: &str| -> Result<u64> {
            let n = s
                .strip_suffix("k prompt tokens)")
                .ok_or_else(|| format!("xAI pricing: bracket {rest:?}"))?;
            n.parse::<u64>()
                .map(|n| n * 1000)
                .map_err(|_| format!("xAI pricing: bracket {rest:?}"))
        };
        if let Some(n) = rest.strip_prefix("< ") {
            below.push((p, k(n)?));
        } else if let Some(n) = rest.strip_prefix("≥ ") {
            above.push((p, k(n)?));
        } else {
            return Err(format!("xAI pricing: bracket {rest:?}"));
        }
    }
    match (&below[..], &above[..]) {
        ([(b, t1)], [(a, t2)]) if t1 == t2 => Ok((*b, *a, *t1)),
        _ => Err(format!(
            "xAI pricing: expected one `<` and one `≥` row for {id:?} with one threshold"
        )),
    }
}

// --- DeepSeek: api-docs.deepseek.com/quick_start/pricing (HTML only) --------------------------

#[derive(Clone, Copy, Debug)]
pub struct DeepSeekModel {
    pub peak: Published,
    pub off_peak: Published,
}

/// The "Model Details" table: the model columns, then the six price rows (cache hit, cache miss,
/// output; off-peak and peak each).
pub fn deepseek(article: &str) -> Result<BTreeMap<String, DeepSeekModel>> {
    let rows = html_rows(article);
    let header = one(
        rows.iter()
            .filter(|r| r.first().map(String::as_str) == Some("MODEL"))
            .collect(),
        "DeepSeek pricing: the MODEL row",
    )?;
    let models: Vec<String> = header[1..]
        .iter()
        .map(|m| m.split('(').next().unwrap_or(m).trim().to_owned())
        .collect();
    if models.is_empty() {
        return Err("DeepSeek pricing: no model columns".into());
    }
    // (kind, window) → one price per model.
    let mut prices: BTreeMap<(String, String), Vec<Rate>> = BTreeMap::new();
    let mut kind: Option<String> = None;
    let mut in_pricing = false;
    for r in &rows {
        let mut cells: &[String] = r;
        if cells.first().is_some_and(|c| c.starts_with("PRICING")) {
            in_pricing = true;
            cells = &cells[1..];
        }
        if !in_pricing {
            continue;
        }
        if cells.first().is_some_and(|c| c.starts_with("1M ")) {
            kind = Some(cells[0].replace(' ', ""));
            cells = &cells[1..];
        }
        let Some(window) = cells.first().filter(|c| *c == "OFF-PEAK" || *c == "PEAK") else {
            // The first row past the price rows ends the pricing block.
            if prices.len() == 6 {
                break;
            }
            return Err(format!(
                "DeepSeek pricing: unexpected row in the price block: {r:?}"
            ));
        };
        let k = kind
            .clone()
            .ok_or("DeepSeek pricing: a price row before any `1M …` label")?;
        let vals = cells[1..]
            .iter()
            .map(|c| dollars(c))
            .collect::<Result<Vec<_>>>()?;
        if vals.len() != models.len() {
            return Err(format!(
                "DeepSeek pricing: {k} {window}: {} prices for {} models",
                vals.len(),
                models.len()
            ));
        }
        if prices.insert((k.clone(), window.clone()), vals).is_some() {
            return Err(format!("DeepSeek pricing: {k} {window} twice"));
        }
    }
    let get = |k: &str, w: &str, i: usize| -> Result<Rate> {
        prices
            .get(&(k.to_owned(), w.to_owned()))
            .and_then(|v| v.get(i).copied())
            .ok_or_else(|| format!("DeepSeek pricing: no {k} {w} row"))
    };
    let mut out = BTreeMap::new();
    for (i, m) in models.iter().enumerate() {
        let tier = |w: &str| -> Result<Published> {
            Ok(Published {
                input: get("1MINPUTTOKENS(CACHEMISS)", w, i)?,
                cache_read: Some(get("1MINPUTTOKENS(CACHEHIT)", w, i)?),
                output: Some(get("1MOUTPUTTOKENS", w, i)?),
                cache_write: None,
                cache_write_1h: None,
            })
        };
        out.insert(
            m.clone(),
            DeepSeekModel {
                peak: tier("PEAK")?,
                off_peak: tier("OFF-PEAK")?,
            },
        );
    }
    if prices.len() != 6 {
        return Err(format!(
            "DeepSeek pricing: {} price rows, expected 6",
            prices.len()
        ));
    }
    Ok(out)
}

// --- Groq: console.groq.com/docs/models.md ----------------------------------------------------

const GROQ_HEADER: &[&str] = &[
    "MODEL ID",
    "SPEED (T/SEC)",
    "PRICE PER 1M TOKENS",
    "RATE LIMITS (DEVELOPER PLAN)",
    "CONTEXT WINDOW (TOKENS)",
    "MAX COMPLETION TOKENS",
    "MAX FILE SIZE",
];

/// (input, output) for a model id.
pub fn groq(md: &str, id: &str) -> Result<(Rate, Rate)> {
    let ts = tables(md)?;
    let model_tables: Vec<&Table> = ts
        .iter()
        .filter(|t| t.header.first().map(String::as_str) == Some("MODEL ID"))
        .collect();
    if model_tables.is_empty() {
        return Err("Groq models: no model tables".into());
    }
    for t in &model_tables {
        t.expect_header(GROQ_HEADER, "Groq models table")?;
    }
    let suffix = format!("){id}");
    let r = one(
        model_tables
            .iter()
            .flat_map(|t| t.rows.iter())
            .filter(|r| r[0].ends_with(&suffix))
            .collect(),
        &format!("Groq model {id:?}"),
    )?;
    let p = &r[2];
    let (i, o) = p
        .strip_suffix(" output")
        .and_then(|p| p.split_once(" input"))
        .ok_or_else(|| format!("Groq {id}: price {p:?} is not `$a input$b output`"))?;
    Ok((dollars(i)?, dollars(o)?))
}

/// Whether the prompt-caching page's Supported Models table lists the model id.
pub fn groq_caches(md: &str, id: &str) -> Result<bool> {
    let ts = tables(md)?;
    let t = one(
        ts.iter()
            .filter(|t| {
                t.context
                    .iter()
                    .any(|c| c == "## [Supported Models](#supported-models)")
            })
            .collect(),
        "Groq prompt caching: the supported models table",
    )?;
    t.expect_header(&["Model ID", "Model"], "Groq prompt caching table")?;
    Ok(t.rows.iter().any(|r| r[0] == id))
}

// --- Fireworks: docs.fireworks.ai/serverless/pricing.md ---------------------------------------

/// (standard, priority) for the row named `name` (its link text) whose link is the model's page
/// `…/models/fireworks/{slug}`.
pub fn fireworks(md: &str, name: &str, slug: &str) -> Result<(Published, Option<Published>)> {
    let ts = tables(md)?;
    let t = one(
        ts.iter()
            .filter(|t| {
                t.header
                    .iter()
                    .map(String::as_str)
                    .eq(["Model", "Standard", "Priority"])
            })
            .collect(),
        "Fireworks pricing: the text and vision models table",
    )?;
    let want = format!("[{name}](https://app.fireworks.ai/models/fireworks/{slug})");
    let r = one(
        t.rows.iter().filter(|r| r[0] == want).collect(),
        &format!("Fireworks row {want}"),
    )?;
    let triple = |cell: &str| -> Result<Published> {
        let parts: Vec<&str> = cell.split(" / ").collect();
        let [i, c, o] = parts[..] else {
            return Err(format!(
                "Fireworks {name}: {cell:?} is not `in / cached / out`"
            ));
        };
        Ok(Published {
            input: dollars(i)?,
            cache_read: Some(dollars(c)?),
            output: Some(dollars(o)?),
            cache_write: None,
            cache_write_1h: None,
        })
    };
    let priority = match r[2].trim() {
        "—" => None,
        c => Some(triple(c)?),
    };
    Ok((triple(&r[1])?, priority))
}

// --- Together: GET /v1/models, cross-checked with docs.together.ai/docs/serverless/models.md ---

/// (input, output, cached input) from the models API.
pub fn together_api(json: &str, id: &str) -> Result<(Rate, Rate, Option<Rate>)> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("Together models: {e}"))?;
    let all = v.as_array().ok_or("Together models: not an array")?;
    let m = one(
        all.iter()
            .filter(|m| m.get("id").and_then(Value::as_str) == Some(id))
            .collect(),
        &format!("Together model {id:?}"),
    )?;
    let p = m
        .get("pricing")
        .ok_or_else(|| format!("Together {id}: no pricing"))?;
    let f = |k: &str| -> Result<Option<Rate>> {
        match p.get(k) {
            None => Ok(None),
            Some(x) => {
                let x = x
                    .as_f64()
                    .ok_or_else(|| format!("Together {id}: pricing.{k} is not a number"))?;
                float_per_million(x)
                    .map(Some)
                    .map_err(|e| format!("Together {id}: {e}"))
            }
        }
    };
    let input = f("input")?.ok_or_else(|| format!("Together {id}: no input price"))?;
    let output = f("output")?.ok_or_else(|| format!("Together {id}: no output price"))?;
    if input == Rate::ZERO {
        return Err(format!("Together {id}: a zero input price"));
    }
    Ok((input, output, f("cached_input")?))
}

const TOGETHER_DOCS_HEADER: &[&str] = &[
    "Organization",
    "Model name",
    "API model string",
    "Context length",
    "Input pricing (per 1M tokens)",
    "Cached input pricing (per 1M tokens)",
    "Output pricing (per 1M tokens)",
    "Quantization",
    "Function calling",
    "Structured outputs",
];

/// (input, output, cached input) from the docs' chat-models table, or `None` if it does not list
/// the model.
pub fn together_docs(md: &str, id: &str) -> Result<Option<(Rate, Rate, Option<Rate>)>> {
    let ts = tables(md)?;
    let t = one(
        ts.iter()
            .filter(|t| {
                t.header
                    .iter()
                    .map(String::as_str)
                    .eq(TOGETHER_DOCS_HEADER.iter().copied())
            })
            .collect(),
        "Together docs: the chat models table",
    )?;
    let hits: Vec<&Vec<String>> = t.rows.iter().filter(|r| r[2] == id).collect();
    match hits[..] {
        [] => Ok(None),
        [r] => Ok(Some((
            dollars(&r[4])?,
            dollars(&r[6])?,
            opt_dollars(&r[5])?,
        ))),
        _ => Err(format!("Together docs: {id} listed twice")),
    }
}

// --- Amazon Bedrock: the AWS Price List bulk offer ---------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BedrockScope {
    pub input: Option<Rate>,
    pub output: Option<Rate>,
    pub cache_read: Option<Rate>,
    pub cache_write: Option<Rate>,
    pub cache_write_1h: Option<Rate>,
}

/// The on-demand per-token SKUs for one model (`attributes.model`), split into the Regional
/// (geo-profile, `us.`) and Global scopes. Batch, reserved-throughput and latency SKUs are
/// skipped by name; any other usage type on the model is an error.
pub fn bedrock(json: &str, model: &str) -> Result<(BedrockScope, BedrockScope)> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("AWS offer: {e}"))?;
    let products = v
        .get("products")
        .and_then(Value::as_object)
        .ok_or("AWS offer: no products")?;
    let terms = v
        .pointer("/terms/OnDemand")
        .and_then(Value::as_object)
        .ok_or("AWS offer: no OnDemand terms")?;
    let (mut regional, mut global) = (BedrockScope::default(), BedrockScope::default());
    let mut seen = 0;
    for (sku, p) in products {
        if p.pointer("/attributes/model").and_then(Value::as_str) != Some(model) {
            continue;
        }
        let ut = p
            .pointer("/attributes/usagetype")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("AWS {model}: SKU {sku} has no usagetype"))?;
        let u = ut.to_ascii_lowercase();
        if [
            "batch",
            "reserved",
            "tpm",
            "latency",
            "flex",
            "priority",
            "provisioned",
        ]
        .iter()
        .any(|k| u.contains(k))
        {
            continue;
        }
        let field = if u.contains("cacheread") || u.contains("cache_read") {
            "cache_read"
        } else if u.contains("cachewrite1h") || u.contains("cache_write_tokens_1h") {
            "cache_write_1h"
        } else if u.contains("cachewrite") || u.contains("cache_write") {
            "cache_write"
        } else if u.contains("inputtoken") || u.contains("input_tokens") {
            "input"
        } else if u.contains("outputtoken") || u.contains("output_tokens") {
            "output"
        } else {
            return Err(format!("AWS {model}: unrecognized usage type {ut}"));
        };
        let scope = if u.contains("global") {
            &mut global
        } else {
            &mut regional
        };
        let offers = terms
            .get(sku)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("AWS {model}: SKU {sku} has no on-demand term"))?;
        let mut prices = Vec::new();
        for o in offers.values() {
            for d in o
                .get("priceDimensions")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|m| m.values())
            {
                let unit = d.get("unit").and_then(Value::as_str).unwrap_or("");
                if unit != "1M tokens" {
                    return Err(format!("AWS {model}: {ut} is priced per {unit:?}"));
                }
                let usd = d
                    .pointer("/pricePerUnit/USD")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("AWS {model}: {ut} has no USD price"))?;
                prices.push(Rate::per_million(usd)?);
            }
        }
        let [price] = prices[..] else {
            return Err(format!("AWS {model}: {ut} has {} prices", prices.len()));
        };
        let slot = match field {
            "input" => &mut scope.input,
            "output" => &mut scope.output,
            "cache_read" => &mut scope.cache_read,
            "cache_write" => &mut scope.cache_write,
            _ => &mut scope.cache_write_1h,
        };
        if slot.replace(price).is_some() {
            return Err(format!("AWS {model}: two {field} SKUs in one scope ({ut})"));
        }
        seen += 1;
    }
    if seen == 0 {
        return Err(format!("AWS offer: no on-demand token SKUs for {model:?}"));
    }
    Ok((regional, global))
}

// --- OpenRouter: GET /api/v1/models/{slug}/endpoints -------------------------------------------

/// One endpoint's prices: the base rates, the long-context override, and the time-of-day
/// schedule folded to its dearest window.
#[derive(Clone, Debug)]
pub struct OrEndpointSrc {
    pub host: String,
    pub tag: String,
    pub standard: Published,
    /// `(min_prompt_tokens, rates)`.
    pub long: Option<(u64, Published)>,
    pub web_search: Option<Rate>,
    pub time_of_day: bool,
}

/// Pricing keys that carry no charge the pricer must know about when zero, and are an error
/// when not (a per-request or per-image fee the table cannot hold).
const OR_ZERO_ONLY: &[&str] = &[
    "request",
    "image",
    "image_output",
    "image_token",
    "input_audio",
    "audio",
    "internal_reasoning",
    "input_audio_cache",
];

pub fn openrouter(json: &str, slug: &str) -> Result<Vec<OrEndpointSrc>> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("OpenRouter {slug}: {e}"))?;
    let eps = v
        .pointer("/data/endpoints")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("OpenRouter {slug}: no endpoints"))?;
    let mut out = Vec::new();
    for e in eps {
        let host = crate::json::str_at(e, "provider_name", slug)?.to_owned();
        let tag = crate::json::str_at(e, "tag", slug)?.to_owned();
        let what = format!("OpenRouter {slug} {tag}");
        let p = e
            .get("pricing")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{what}: no pricing"))?;
        let tok = |m: &serde_json::Map<String, Value>, k: &str| -> Result<Option<Rate>> {
            match m.get(k) {
                None => Ok(None),
                Some(x) => {
                    let s = x
                        .as_str()
                        .ok_or_else(|| format!("{what}: {k} is not a string"))?;
                    Rate::per_token(s)
                        .map(Some)
                        .map_err(|e| format!("{what}: {k}: {e}"))
                }
            }
        };
        for (k, x) in p {
            let known = [
                "prompt",
                "completion",
                "input_cache_read",
                "input_cache_write",
                "input_cache_write_1h",
                "web_search",
                "discount",
                "overrides",
            ];
            if known.contains(&k.as_str()) {
                continue;
            }
            if !OR_ZERO_ONLY.contains(&k.as_str()) {
                return Err(format!("{what}: unknown pricing key {k:?}"));
            }
            let zero = x
                .as_str()
                .is_some_and(|s| Rate::per_token(s) == Ok(Rate::ZERO));
            if !zero {
                return Err(format!(
                    "{what}: a nonzero {k} fee ({x}) the table cannot hold"
                ));
            }
        }
        let base = |m: &serde_json::Map<String, Value>,
                    inherit: Option<&Published>|
         -> Result<Published> {
            let input = match (tok(m, "prompt")?, inherit) {
                (Some(r), _) => r,
                (None, Some(b)) => b.input,
                (None, None) => return Err(format!("{what}: no prompt price")),
            };
            Ok(Published {
                input,
                output: tok(m, "completion")?.or(inherit.and_then(|b| b.output)),
                cache_read: tok(m, "input_cache_read")?,
                cache_write: tok(m, "input_cache_write")?,
                cache_write_1h: tok(m, "input_cache_write_1h")?,
            })
        };
        let mut standard = base(p, None)?;
        if standard.output.is_none() {
            return Err(format!("{what}: no completion price"));
        }
        let mut long = None;
        let mut time_of_day = false;
        for o in p
            .get("overrides")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
        {
            let o = o
                .as_object()
                .ok_or_else(|| format!("{what}: an override is not an object"))?;
            let timed = o.keys().any(|k| k.starts_with("utc_"));
            let min = o.get("min_prompt_tokens").and_then(Value::as_u64);
            for k in o.keys() {
                let ok = k.starts_with("utc_")
                    || k == "min_prompt_tokens"
                    || [
                        "prompt",
                        "completion",
                        "input_cache_read",
                        "input_cache_write",
                        "input_cache_write_1h",
                    ]
                    .contains(&k.as_str());
                if !ok {
                    return Err(format!("{what}: unknown override key {k:?}"));
                }
            }
            match (timed, min) {
                (true, None) => {
                    // A time-of-day window: fold to the dearest window, field by field.
                    time_of_day = true;
                    let w = base(o, Some(&standard))?;
                    standard.input = standard.input.max(w.input);
                    standard.output = standard.output.max(w.output);
                    standard.cache_read = max_opt(standard.cache_read, w.cache_read);
                    standard.cache_write = max_opt(standard.cache_write, w.cache_write);
                    standard.cache_write_1h = max_opt(standard.cache_write_1h, w.cache_write_1h);
                }
                (false, Some(n)) => {
                    if long.is_some() {
                        return Err(format!("{what}: two long-context overrides"));
                    }
                    long = Some((n, base(o, Some(&standard))?));
                }
                _ => {
                    return Err(format!(
                        "{what}: an override that is neither a window nor a tier"
                    ));
                }
            }
        }
        if time_of_day && long.is_some() {
            return Err(format!(
                "{what}: a time-of-day schedule and a long tier together"
            ));
        }
        out.push(OrEndpointSrc {
            host,
            tag,
            standard,
            long,
            // USD per search, not per token.
            web_search: match p.get("web_search").and_then(Value::as_str) {
                None => None,
                Some(s) => {
                    Some(Rate::per_million(s).map_err(|e| format!("{what}: web_search: {e}"))?)
                }
            }
            .filter(|r| *r != Rate::ZERO),
            time_of_day,
        });
    }
    if out.is_empty() {
        return Err(format!("OpenRouter {slug}: no endpoints"));
    }
    Ok(out)
}

fn max_opt(a: Option<Rate>, b: Option<Rate>) -> Option<Rate> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// OpenRouter's model list names the slug (it is a model OpenRouter serves).
pub fn openrouter_listed(json: &str, slug: &str) -> Result<()> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("OpenRouter models: {e}"))?;
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or("OpenRouter models: no data")?;
    one(
        data.iter()
            .filter(|m| m.get("id").and_then(Value::as_str) == Some(slug))
            .collect(),
        &format!("OpenRouter model list: {slug}"),
    )
    .map(|_| ())
}

/// Whitespace-collapsed plain text of a snapshot, for a prose quote's check.
pub fn plain_text(snapshot: &str, html: bool) -> String {
    if html {
        md::html_text(snapshot)
    } else {
        collapse(snapshot)
    }
}
