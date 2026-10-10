//! The hand-entered half of the rate table, read from `verify/catalog_truth.toml`: which source
//! row each card is, the dimensions that live only in a vendor's prose (each with the URL and the
//! verbatim quote that states it), per-call tool fees, recorded source conflicts and overrides,
//! and which card each catalog row's candidates bill from (`[[pricing]]`).

use crate::Result;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Prose {
    pub id: String,
    pub value: String,
    pub url: String,
    pub quote: String,
}

#[derive(Clone, Debug)]
pub struct Fee {
    pub tool: String,
    pub usd: String,
    pub quote: String,
}

#[derive(Clone, Debug)]
pub struct Tools {
    pub id: String,
    pub url: String,
    pub fees: Vec<Fee>,
}

#[derive(Clone, Debug)]
pub struct CardSpec {
    pub name: String,
    /// The vendor's key for the row: a table's model name, an API id or alias.
    pub row: String,
    pub geo_us: Option<String>,
    pub tools: Option<String>,
    /// OpenAI embeddings: the output rate where the table publishes none.
    pub unpublished_output: Option<String>,
}

/// Two machine-readable sources disagree on one field, and this records which one the table
/// follows and why. The recorded values must still be what the sources say, or it is stale.
#[derive(Clone, Debug)]
pub struct Conflict {
    pub card: String,
    pub field: String,
    pub values: BTreeMap<String, String>,
    pub follow: String,
    pub reason: String,
}

/// The table holds a rate other than the source's (a promotion ending before the next sync).
/// `source_says` must be what the source says, or the override is stale and generation fails.
#[derive(Clone, Debug)]
pub struct Override {
    pub card: String,
    pub source_says: BTreeMap<String, String>,
    pub holds: BTreeMap<String, String>,
    pub until: String,
    pub url: String,
    pub quote: String,
}

/// A rate the vendor billed until `until` (exclusive) and has since changed: the invoice check
/// (`rates-sync audit`) accepts a line from before then at this rate. Never used to generate.
#[derive(Clone, Debug)]
pub struct Superseded {
    pub card: String,
    pub field: String,
    pub usd: String,
    pub until: String,
    pub url: String,
    pub quote: String,
}

#[derive(Clone, Debug)]
pub struct Calendar {
    pub holidays: Vec<String>,
    pub covered: (String, String),
    pub url: String,
    pub quote: String,
}

#[derive(Clone, Debug)]
pub struct Pricing {
    pub model: String,
    pub list_card: String,
    /// Provider name → `list`, a card name, `openrouter:{slug}`, or `unverified: {reason}`.
    pub cost: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub prose: BTreeMap<String, Prose>,
    pub tools: Vec<Tools>,
    pub cards: Vec<CardSpec>,
    pub conflicts: Vec<Conflict>,
    pub overrides: Vec<Override>,
    pub superseded: Vec<Superseded>,
    pub calendar: Calendar,
    pub pricing: Vec<Pricing>,
}

fn arr<'a>(t: &'a toml::Table, k: &str) -> &'a [toml::Value] {
    t.get(k)
        .and_then(toml::Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn s(t: &toml::Table, k: &str, what: &str) -> Result<String> {
    t.get(k)
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("catalog_truth.toml: {what} lacks `{k}`"))
}

fn opt(t: &toml::Table, k: &str) -> Option<String> {
    t.get(k).and_then(toml::Value::as_str).map(str::to_owned)
}

fn table<'a>(v: &'a toml::Value, what: &str) -> Result<&'a toml::Table> {
    v.as_table()
        .ok_or_else(|| format!("catalog_truth.toml: {what} is not a table"))
}

fn str_map(t: &toml::Table, k: &str, what: &str) -> Result<BTreeMap<String, String>> {
    let m = t
        .get(k)
        .and_then(toml::Value::as_table)
        .ok_or_else(|| format!("catalog_truth.toml: {what} lacks table `{k}`"))?;
    m.iter()
        .map(|(k2, v)| {
            v.as_str()
                .map(|x| (k2.clone(), x.to_owned()))
                .ok_or_else(|| format!("catalog_truth.toml: {what}.{k}.{k2} is not a string"))
        })
        .collect()
}

pub fn parse(text: &str) -> Result<Spec> {
    let t: toml::Table = text
        .parse()
        .map_err(|e| format!("catalog_truth.toml: {e}"))?;
    let mut prose = BTreeMap::new();
    for p in arr(&t, "prose") {
        let p = table(p, "[[prose]]")?;
        let id = s(p, "id", "[[prose]]")?;
        let e = Prose {
            value: s(p, "value", &id)?,
            url: s(p, "url", &id)?,
            quote: s(p, "quote", &id)?,
            id: id.clone(),
        };
        if prose.insert(id.clone(), e).is_some() {
            return Err(format!("catalog_truth.toml: [[prose]] {id} twice"));
        }
    }
    let mut tools = Vec::new();
    for x in arr(&t, "tools") {
        let x = table(x, "[[tools]]")?;
        let id = s(x, "id", "[[tools]]")?;
        let mut fees = Vec::new();
        for f in arr(x, "fees") {
            let f = table(f, &id)?;
            fees.push(Fee {
                tool: s(f, "tool", &id)?,
                usd: s(f, "usd", &id)?,
                quote: s(f, "quote", &id)?,
            });
        }
        tools.push(Tools {
            url: s(x, "url", &id)?,
            id,
            fees,
        });
    }
    let mut cards = Vec::new();
    for c in arr(&t, "card") {
        let c = table(c, "[[card]]")?;
        let name = s(c, "name", "[[card]]")?;
        cards.push(CardSpec {
            row: s(c, "row", &name)?,
            geo_us: opt(c, "geo_us"),
            tools: opt(c, "tools"),
            unpublished_output: opt(c, "unpublished_output"),
            name,
        });
    }
    let mut conflicts = Vec::new();
    for c in arr(&t, "conflict") {
        let c = table(c, "[[conflict]]")?;
        let card = s(c, "card", "[[conflict]]")?;
        conflicts.push(Conflict {
            field: s(c, "field", &card)?,
            values: str_map(c, "values", &card)?,
            follow: s(c, "follow", &card)?,
            reason: s(c, "reason", &card)?,
            card,
        });
    }
    let mut overrides = Vec::new();
    for o in arr(&t, "override") {
        let o = table(o, "[[override]]")?;
        let card = s(o, "card", "[[override]]")?;
        overrides.push(Override {
            source_says: str_map(o, "source_says", &card)?,
            holds: str_map(o, "holds", &card)?,
            until: s(o, "until", &card)?,
            url: s(o, "url", &card)?,
            quote: s(o, "quote", &card)?,
            card,
        });
    }
    let mut superseded = Vec::new();
    for x in arr(&t, "superseded") {
        let x = table(x, "[[superseded]]")?;
        let card = s(x, "card", "[[superseded]]")?;
        superseded.push(Superseded {
            field: s(x, "field", &card)?,
            usd: s(x, "usd", &card)?,
            until: s(x, "until", &card)?,
            url: s(x, "url", &card)?,
            quote: s(x, "quote", &card)?,
            card,
        });
    }
    let cal = t
        .get("deepseek_calendar")
        .and_then(toml::Value::as_table)
        .ok_or("catalog_truth.toml: no [deepseek_calendar]")?;
    let strs = |k: &str| -> Result<Vec<String>> {
        arr(cal, k)
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("[deepseek_calendar].{k}: not a string"))
            })
            .collect()
    };
    let covered = strs("covered")?;
    let [from, until] = &covered[..] else {
        return Err("[deepseek_calendar].covered: expected [from, until]".into());
    };
    let calendar = Calendar {
        holidays: strs("holidays")?,
        covered: (from.clone(), until.clone()),
        url: s(cal, "url", "[deepseek_calendar]")?,
        quote: s(cal, "quote", "[deepseek_calendar]")?,
    };
    let mut pricing = Vec::new();
    for p in arr(&t, "pricing") {
        let p = table(p, "[[pricing]]")?;
        let model = s(p, "model", "[[pricing]]")?;
        pricing.push(Pricing {
            list_card: s(p, "list_card", &model)?,
            cost: str_map(p, "cost", &model)?,
            model,
        });
    }
    Ok(Spec {
        prose,
        tools,
        cards,
        conflicts,
        overrides,
        superseded,
        calendar,
        pricing,
    })
}

impl Spec {
    pub fn prose(&self, id: &str) -> Result<&Prose> {
        self.prose
            .get(id)
            .ok_or_else(|| format!("catalog_truth.toml: no [[prose]] {id}"))
    }

    /// Every OpenRouter slug a row bills from.
    pub fn openrouter_slugs(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .pricing
            .iter()
            .flat_map(|p| p.cost.values())
            .filter_map(|c| c.strip_prefix("openrouter:"))
            .map(str::to_owned)
            .collect();
        v.sort();
        v.dedup();
        v
    }
}
