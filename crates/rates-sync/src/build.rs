//! Assembling the table: each card from its vendor's source row plus the cited prose rules, every
//! cross-check between two sources of the same fact, and the per-row cost wiring from
//! `[[pricing]]` and the catalog.

use crate::Result;
use crate::dec::{Rate, bps, decimal};
use crate::snapshot::Store;
use crate::spec::{CardSpec, Spec};
use crate::vendors::{self, Published};
use providers::catalog::MODEL_ROUTES;
use std::collections::BTreeMap;

/// One rate set, resolved: every field the pricer needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tr {
    pub input: Rate,
    pub output: Rate,
    pub cache_read: Rate,
    pub cache_write_5m: Rate,
    pub cache_write_1h: Option<Rate>,
}

impl Tr {
    /// The table's rule for an unpublished rate: a cache read or write the vendor does not price
    /// apart is billed at the input rate (no discount, no premium; never $0).
    fn from(p: &Published, what: &str) -> Result<Tr> {
        Ok(Tr {
            input: p.input,
            output: p.output.ok_or_else(|| format!("{what}: no output rate"))?,
            cache_read: p.cache_read.unwrap_or(p.input),
            cache_write_5m: p.cache_write.unwrap_or(p.input),
            cache_write_1h: p.cache_write_1h,
        })
    }

    fn scale(&self, num: u128, den: u128) -> Result<Tr> {
        Ok(Tr {
            input: self.input.ratio(num, den)?,
            output: self.output.ratio(num, den)?,
            cache_read: self.cache_read.ratio(num, den)?,
            cache_write_5m: self.cache_write_5m.ratio(num, den)?,
            cache_write_1h: self.cache_write_1h.map(|r| r.ratio(num, den)).transpose()?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffPeak {
    pub peak: Vec<(u16, u16)>,
    pub weekdays_only: bool,
    pub holidays: Vec<u32>,
    pub covered_from: u32,
    pub covered_until: u32,
    pub rates: Tr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    pub standard: Tr,
    pub long: Option<Tr>,
    pub fast: Option<Tr>,
    pub fast_long: Option<Tr>,
    pub ultrafast: Option<Tr>,
    pub ultrafast_long: Option<Tr>,
    pub flex: Option<Tr>,
    pub flex_long: Option<Tr>,
    pub off_peak: Option<OffPeak>,
    pub long_context: Option<(u64, bool)>,
    pub geo_us: Option<u128>,
    /// The fee list's const name.
    pub tools: Option<String>,
}

impl Card {
    fn new(standard: Tr) -> Card {
        Card {
            standard,
            long: None,
            fast: None,
            fast_long: None,
            ultrafast: None,
            ultrafast_long: None,
            flex: None,
            flex_long: None,
            off_peak: None,
            long_context: None,
            geo_us: None,
            tools: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Endpoint {
    pub host: String,
    pub tag: String,
    pub class: &'static str,
    pub card: Card,
}

#[derive(Clone, Debug)]
pub enum Cost {
    List,
    Own(String),
    OpenRouter(String),
    Unverified(String),
}

#[derive(Clone, Debug)]
pub enum List {
    Card(String),
    Inline(Tr),
}

#[derive(Clone, Debug)]
pub struct Row {
    pub model: String,
    pub list: List,
    /// `(ProviderId variant, cost)` in the catalog's candidate order.
    pub cost: Vec<(String, Cost)>,
}

/// The whole generated table.
#[derive(Clone, Debug)]
pub struct Table {
    pub credit_fee_bps: u128,
    /// `(const name, [(Tool variant, micro-dollars)])`.
    pub tools: Vec<(String, Vec<(String, u64)>)>,
    /// `(card name, card)`, in `[[card]]` order.
    pub cards: Vec<(String, Card)>,
    /// `(slug, endpoints)`, sorted by slug.
    pub openrouter: Vec<(String, Vec<Endpoint>)>,
    pub rows: Vec<Row>,
}

/// `anthropic:claude-opus-4-8` → `ANTHROPIC_CLAUDE_OPUS_4_8`.
pub fn const_name(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_uppercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_owned()
}

/// `web_search` → `WebSearch`.
fn camel(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next()
                .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        })
        .collect()
}

/// Days since 1970-01-01 for `YYYY-MM-DD`.
pub fn day(date: &str) -> Result<u32> {
    let p: Vec<i64> = date
        .split('-')
        .map(|x| x.parse::<i64>().map_err(|_| format!("bad date {date:?}")))
        .collect::<Result<_>>()?;
    let [y, m, d] = p[..] else {
        return Err(format!("bad date {date:?}"));
    };
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    u32::try_from(era * 146_097 + doe - 719_468).map_err(|_| format!("bad date {date:?}"))
}

/// Every quote must appear, word for word, in the snapshot of its URL.
fn check_quote(store: &Store, url: &str, quote: &str, what: &str) -> Result<()> {
    let (entry, text) = store
        .by_url(url)
        .ok_or_else(|| format!("{what}: {url} has no snapshot, so its quote cannot be checked"))?;
    let html = entry.file.ends_with(".html");
    let hay = vendors::plain_text(text, html);
    let needle = vendors::plain_text(quote, false);
    if !hay.contains(&needle) {
        return Err(format!(
            "{what}: the quote is no longer on {url} (snapshot {}): {quote:?}",
            entry.file
        ));
    }
    Ok(())
}

struct Ctx<'a> {
    store: &'a Store,
    spec: &'a Spec,
}

impl Ctx<'_> {
    fn prose_value(&self, id: &str) -> Result<String> {
        let p = self.spec.prose(id)?;
        check_quote(self.store, &p.url, &p.quote, &format!("[[prose]] {id}"))?;
        Ok(p.value.clone())
    }

    fn prose_bps(&self, id: &str) -> Result<u128> {
        bps(&self.prose_value(id)?).map_err(|e| format!("[[prose]] {id}: {e}"))
    }

    /// A recorded conflict for (card, field), checked against what the sources now say: the
    /// value to use.
    fn resolve(
        &self,
        card: &str,
        field: &str,
        says: &[(&str, Option<Rate>)],
    ) -> Result<Option<Rate>> {
        let show = |r: &Option<Rate>| {
            r.map_or("absent".to_owned(), |r| r.table_text().unwrap_or_default())
        };
        let first = says[0].1;
        if says.iter().all(|(_, r)| *r == first) {
            if let Some(c) = self
                .spec
                .conflicts
                .iter()
                .find(|c| c.card == card && c.field == field)
            {
                return Err(format!(
                    "[[conflict]] {card} {field}: the sources now agree ({}); remove it ({})",
                    show(&first),
                    c.reason
                ));
            }
            return Ok(first);
        }
        let listing = says
            .iter()
            .map(|(s, r)| format!("{s} = {}", show(r)))
            .collect::<Vec<_>>()
            .join(", ");
        let c = self
            .spec
            .conflicts
            .iter()
            .find(|c| c.card == card && c.field == field)
            .ok_or_else(|| format!("{card}: the sources disagree on {field}: {listing}. Record a [[conflict]] that says which one to follow and why, or fix the source"))?;
        for (s, r) in says {
            if c.values.get(*s).map(String::as_str) != Some(show(r).as_str()) {
                return Err(format!(
                    "[[conflict]] {card} {field} is stale: the sources now say {listing}"
                ));
            }
        }
        says.iter()
            .find(|(s, _)| *s == c.follow)
            .map(|(_, r)| *r)
            .ok_or_else(|| {
                format!(
                    "[[conflict]] {card} {field}: follows unknown source {}",
                    c.follow
                )
            })
    }

    fn tools(&self, spec: &CardSpec) -> Option<String> {
        spec.tools
            .as_ref()
            .map(|t| format!("{}_TOOLS", const_name(t)))
    }

    fn card(&self, spec: &CardSpec) -> Result<Card> {
        let (vendor, _) = spec
            .name
            .split_once(':')
            .ok_or_else(|| format!("[[card]] {}: name is not vendor:model", spec.name))?;
        let name = spec.name.as_str();
        let mut card = match vendor {
            "anthropic" => self.anthropic(spec)?,
            "bedrock" => self.bedrock(spec)?,
            "openai" => self.openai(spec)?,
            "xai" => self.xai(spec)?,
            "deepseek" => self.deepseek(spec)?,
            "together" => self.together(spec)?,
            "fireworks" => self.fireworks(spec)?,
            "groq" => self.groq(spec)?,
            _ => {
                return Err(format!(
                    "[[card]] {name}: no source reader for vendor {vendor}"
                ));
            }
        };
        if let Some(g) = &spec.geo_us {
            card.geo_us = Some(self.prose_bps(g)?);
        }
        card.tools = self.tools(spec);
        Ok(card)
    }

    fn anthropic(&self, spec: &CardSpec) -> Result<Card> {
        let a = vendors::anthropic(self.store.get("anthropic.pricing")?)?;
        let p = a.model(&spec.row)?;
        let standard = Tr::from(&p, &spec.name)?;
        let mut card = Card::new(standard);
        if let Some((fi, fo)) = a.fast(&spec.row)? {
            // "Prompt caching multipliers apply on top of fast mode pricing": the model's own
            // cache-to-input ratios, applied to the fast input rate.
            self.prose_value("anthropic.fast_caching")?;
            let r = |x: Rate| fi.ratio(x.0, standard.input.0);
            card.fast = Some(Tr {
                input: fi,
                output: fo,
                cache_read: r(standard.cache_read)?,
                cache_write_5m: r(standard.cache_write_5m)?,
                cache_write_1h: standard.cache_write_1h.map(r).transpose()?,
            });
        }
        Ok(card)
    }

    fn bedrock(&self, spec: &CardSpec) -> Result<Card> {
        let (regional, global) = vendors::bedrock(self.store.get("aws.bedrock")?, &spec.row)?;
        let req = |r: Option<Rate>, f: &str, scope: &str| {
            r.ok_or_else(|| format!("AWS {}: no {scope} {f} SKU", spec.row))
        };
        let tr = |s: &vendors::BedrockScope, scope: &str| -> Result<Tr> {
            Ok(Tr {
                input: req(s.input, "input", scope)?,
                output: req(s.output, "output", scope)?,
                cache_read: req(s.cache_read, "cache read", scope)?,
                cache_write_5m: req(s.cache_write, "cache write", scope)?,
                cache_write_1h: Some(req(s.cache_write_1h, "1-hour cache write", scope)?),
            })
        };
        let reg = tr(&regional, "Regional")?;
        let glob = tr(&global, "Global")?;
        // Cross-checks: Global is Anthropic's own list, and Regional is Global plus the premium.
        let list = Tr::from(
            &vendors::anthropic(self.store.get("anthropic.pricing")?)?.model(&spec.row)?,
            &spec.row,
        )?;
        if glob != list {
            return Err(format!(
                "{}: AWS Global SKUs {glob:?} are not Anthropic's list {list:?}",
                spec.name
            ));
        }
        let premium = self.prose_bps("bedrock.regional")?;
        if glob.scale(premium, 10_000)? != reg {
            return Err(format!(
                "{}: AWS Regional SKUs {reg:?} are not Global x {premium} bps",
                spec.name
            ));
        }
        Ok(Card::new(reg))
    }

    fn openai(&self, spec: &CardSpec) -> Result<Card> {
        let o = vendors::openai(self.store.get("openai.pricing")?)?;
        let what = &spec.name;
        let fix = |p: Published| -> Result<Published> {
            match (p.output, &spec.unpublished_output) {
                (Some(_), Some(_)) => Err(format!(
                    "{what}: `unpublished_output`, but the table publishes one"
                )),
                (None, Some(x)) => Ok(Published {
                    output: Some(Rate::per_million(x)?),
                    ..p
                }),
                _ => Ok(p),
            }
        };
        let tier = |t: &str| -> Result<(Option<Tr>, Option<Tr>)> {
            match o.tier(t, &spec.row)? {
                None => Ok((None, None)),
                Some(x) => Ok((
                    Some(Tr::from(&fix(x.short)?, what)?),
                    x.long.map(|l| Tr::from(&fix(l)?, what)).transpose()?,
                )),
            }
        };
        let (standard, long) = tier("standard")?;
        let standard = standard
            .ok_or_else(|| format!("{what}: OpenAI's Standard table does not list {}", spec.row))?;
        let (fast, fast_long) = tier("fast")?;
        let (flex, flex_long) = tier("flex")?;
        let (ultrafast, ultrafast_long) = tier("ultrafast")?;
        let any_long = [long, fast_long, flex_long, ultrafast_long]
            .iter()
            .any(Option::is_some);
        let long_context = if any_long {
            Some(threshold(&self.prose_value("openai.long_context")?)?)
        } else {
            None
        };
        Ok(Card {
            long,
            fast,
            fast_long,
            ultrafast,
            ultrafast_long,
            flex,
            flex_long,
            long_context,
            ..Card::new(standard)
        })
    }

    fn xai(&self, spec: &CardSpec) -> Result<Card> {
        let (id, api) = vendors::xai_api(self.store.get("xai.language_models")?, &spec.row)?;
        let (below, above, page_threshold) =
            vendors::xai_page(self.store.get("xai.pricing")?, &id)?;
        if (below, above, page_threshold) != (api.short, api.long, api.threshold) {
            return Err(format!(
                "{}: xAI's models API ({:?} / {:?} at {}) and pricing page ({below:?} / {above:?} at {page_threshold}) disagree",
                spec.name, api.short, api.long, api.threshold
            ));
        }
        let (n, inclusive) = threshold(&self.prose_value("xai.long_context")?)?;
        if n != api.threshold {
            return Err(format!(
                "{}: [[prose]] xai.long_context says {n}, the API says {}",
                spec.name, api.threshold
            ));
        }
        let standard = Tr::from(&api.short, &spec.name)?;
        let priority = self.prose_bps("xai.priority")?;
        Ok(Card {
            long: Some(Tr::from(&api.long, &spec.name)?),
            fast: Some(standard.scale(priority, 10_000)?),
            long_context: Some((n, inclusive)),
            ..Card::new(standard)
        })
    }

    fn deepseek(&self, spec: &CardSpec) -> Result<Card> {
        let all = vendors::deepseek(self.store.get("deepseek.pricing")?)?;
        let m = all
            .get(&spec.row)
            .ok_or_else(|| format!("DeepSeek pricing: no column {:?}", spec.row))?;
        let peak = self.prose_value("deepseek.peak")?;
        let (days, windows) = peak
            .split_once(' ')
            .ok_or("[[prose]] deepseek.peak: expected `Mon-Fri HH:MM-HH:MM,…`")?;
        let weekdays_only = match days {
            "Mon-Fri" => true,
            "daily" => false,
            d => return Err(format!("[[prose]] deepseek.peak: days {d:?}")),
        };
        let peak = windows
            .split(',')
            .map(|w| {
                let (a, b) = w.split_once('-').ok_or_else(|| format!("window {w:?}"))?;
                Ok((minute(a)?, minute(b)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let cal = &self.spec.calendar;
        check_quote(self.store, &cal.url, &cal.quote, "[deepseek_calendar]")?;
        let off = Tr::from(&m.off_peak, &spec.name)?;
        let standard = Tr::from(&m.peak, &spec.name)?;
        Ok(Card {
            off_peak: Some(OffPeak {
                peak,
                weekdays_only,
                holidays: cal.holidays.iter().map(|d| day(d)).collect::<Result<_>>()?,
                covered_from: day(&cal.covered.0)?,
                covered_until: day(&cal.covered.1)?,
                rates: off,
            }),
            ..Card::new(standard)
        })
    }

    fn together(&self, spec: &CardSpec) -> Result<Card> {
        let (ai, ao, ac) = vendors::together_api(self.store.get("together.models")?, &spec.row)?;
        let docs = vendors::together_docs(self.store.get("together.docs")?, &spec.row)?;
        let name = &spec.name;
        let (input, output, cached) = match docs {
            None => {
                self.resolve(
                    name,
                    "listed",
                    &[
                        ("together.models", Some(Rate::ZERO)),
                        ("together.docs", None),
                    ],
                )?;
                (ai, ao, ac)
            }
            Some((di, dout, dc)) => (
                self.resolve(
                    name,
                    "input",
                    &[("together.models", Some(ai)), ("together.docs", Some(di))],
                )?
                .ok_or_else(|| format!("{name}: no input rate"))?,
                self.resolve(
                    name,
                    "output",
                    &[("together.models", Some(ao)), ("together.docs", Some(dout))],
                )?
                .ok_or_else(|| format!("{name}: no output rate"))?,
                self.resolve(
                    name,
                    "cache_read",
                    &[("together.models", ac), ("together.docs", dc)],
                )?,
            ),
        };
        let mut p = Published {
            input,
            output: Some(output),
            cache_read: cached,
            cache_write: None,
            cache_write_1h: None,
        };
        self.apply_override(name, &mut p, "together.models")?;
        Ok(Card::new(Tr::from(&p, name)?))
    }

    fn apply_override(&self, name: &str, p: &mut Published, source: &str) -> Result<()> {
        let Some(o) = self.spec.overrides.iter().find(|o| o.card == name) else {
            return Ok(());
        };
        check_quote(
            self.store,
            &o.url,
            &o.quote,
            &format!("[[override]] {name}"),
        )?;
        let fetched = &self
            .store
            .entries
            .get(source)
            .ok_or_else(|| format!("no snapshot {source}"))?
            .fetched;
        if fetched.as_str() > o.until.as_str() {
            return Err(format!(
                "[[override]] {name} ran until {}; the {source} snapshot is from {fetched}: remove it",
                o.until
            ));
        }
        for (field, want) in &o.source_says {
            let got = field_of(p, field)?
                .map_or("absent".to_owned(), |r| r.table_text().unwrap_or_default());
            if &got != want {
                return Err(format!(
                    "[[override]] {name} is stale: {source} now says {field} = {got}, not {want}: remove it"
                ));
            }
        }
        for (field, v) in &o.holds {
            let r = Rate::per_million(v)?;
            match field.as_str() {
                "input" => p.input = r,
                "output" => p.output = Some(r),
                "cache_read" => p.cache_read = Some(r),
                f => return Err(format!("[[override]] {name}: unknown field {f}")),
            }
        }
        Ok(())
    }

    fn fireworks(&self, spec: &CardSpec) -> Result<Card> {
        let slug = spec.name.rsplit('/').next().unwrap_or("");
        let (std, priority) =
            vendors::fireworks(self.store.get("fireworks.pricing")?, &spec.row, slug)?;
        let fast = match priority {
            Some(p) => {
                self.prose_value("fireworks.priority")?;
                Some(Tr::from(&p, &spec.name)?)
            }
            None => None,
        };
        Ok(Card {
            fast,
            ..Card::new(Tr::from(&std, &spec.name)?)
        })
    }

    fn groq(&self, spec: &CardSpec) -> Result<Card> {
        let (input, output) = vendors::groq(self.store.get("groq.models")?, &spec.row)?;
        let cache_read = if vendors::groq_caches(self.store.get("groq.prompt_caching")?, &spec.row)?
        {
            let d = self.prose_bps("groq.cached")?;
            Some(input.ratio(d, 10_000)?)
        } else {
            None
        };
        let standard = Tr::from(
            &Published {
                input,
                output: Some(output),
                cache_read,
                cache_write: None,
                cache_write_1h: None,
            },
            &spec.name,
        )?;
        // "Flex … pricing matches the on-demand tier."
        self.prose_value("groq.flex")?;
        Ok(Card {
            flex: Some(standard),
            ..Card::new(standard)
        })
    }

    fn openrouter(&self, slug: &str) -> Result<Vec<(Endpoint, Option<u64>)>> {
        let id = format!("openrouter.endpoints/{slug}");
        vendors::openrouter_listed(self.store.get("openrouter.models")?, slug)?;
        let eps = vendors::openrouter(self.store.get(&id)?, slug)?;
        let mut out = Vec::new();
        for e in eps {
            let what = format!("OpenRouter {slug} {}", e.tag);
            let class = match e.tag.rsplit('/').next().unwrap_or("") {
                "fast" | "priority" => "Fast",
                "flex" => "Flex",
                "ultrafast" => "Ultrafast",
                _ => "Standard",
            };
            let mut card = Card::new(Tr::from(&e.standard, &what)?);
            if let Some((n, l)) = &e.long {
                card.long = Some(Tr::from(l, &what)?);
                card.long_context = Some((*n, false));
            }
            let fee = e.web_search.map(fee_micros).transpose()?;
            card.tools = fee.map(or_search_name);
            out.push((
                Endpoint {
                    host: e.host,
                    tag: e.tag,
                    class,
                    card,
                },
                fee,
            ));
        }
        Ok(out)
    }
}

fn field_of(p: &Published, field: &str) -> Result<Option<Rate>> {
    match field {
        "input" => Ok(Some(p.input)),
        "output" => Ok(p.output),
        "cache_read" => Ok(p.cache_read),
        f => Err(format!("unknown field {f}")),
    }
}

/// `>272000` (strictly more) or `>=200000` (reaches).
fn threshold(v: &str) -> Result<(u64, bool)> {
    let (inclusive, n) = match v.strip_prefix(">=") {
        Some(n) => (true, n),
        None => (
            false,
            v.strip_prefix('>')
                .ok_or_else(|| format!("threshold {v:?}: expected `>N` or `>=N`"))?,
        ),
    };
    let n = n.parse().map_err(|_| format!("threshold {v:?}"))?;
    Ok((n, inclusive))
}

fn minute(hhmm: &str) -> Result<u16> {
    let (h, m) = hhmm
        .split_once(':')
        .ok_or_else(|| format!("time {hhmm:?}"))?;
    let h: u16 = h.parse().map_err(|_| format!("time {hhmm:?}"))?;
    let m: u16 = m.parse().map_err(|_| format!("time {hhmm:?}"))?;
    Ok(h * 60 + m)
}

/// A USD-per-call amount (held as a [`Rate`] of the same decimal) in micro-dollars.
fn fee_micros(r: Rate) -> Result<u64> {
    let unit = 1_000_000u128; // Rate units per micro-dollar.
    if !r.0.is_multiple_of(unit) {
        return Err(format!("fee {r} has more than six decimal places"));
    }
    u64::try_from(r.0 / unit).map_err(|_| format!("fee {r} too large"))
}

pub fn or_search_name(micros: u64) -> String {
    if micros.is_multiple_of(1000) {
        format!("OR_SEARCH_{}", micros / 1000)
    } else {
        format!("OR_SEARCH_{micros}U")
    }
}

/// One card from its `[[card]]` spec, as [`build`] reads it: `catalog-drift` prices a row this way
/// before the row exists.
pub fn card(store: &Store, spec: &Spec, c: &CardSpec) -> Result<Card> {
    Ctx { store, spec }.card(c)
}

/// One OpenRouter slug's endpoint cards, as [`build`] reads them (the slug must be in the store's
/// `openrouter.models` and have its `openrouter.endpoints/{slug}` snapshot).
pub fn openrouter(store: &Store, spec: &Spec, slug: &str) -> Result<Vec<Endpoint>> {
    let eps = Ctx { store, spec }.openrouter(slug)?;
    Ok(eps.into_iter().map(|(e, _)| e).collect())
}

/// Build the table from the snapshots and the spec.
pub fn build(store: &Store, spec: &Spec) -> Result<Table> {
    let cx = Ctx { store, spec };
    let fee = cx.prose_value("openrouter.credit_fee")?;
    let credit_fee_bps = 10_000 + bps(&fee)?;

    let mut tools = Vec::new();
    for t in &spec.tools {
        let mut fees = Vec::new();
        for f in &t.fees {
            if providers::pricing::Tool::parse(&f.tool).is_none() {
                return Err(format!("[[tools]] {}: unknown tool {}", t.id, f.tool));
            }
            check_quote(
                store,
                &t.url,
                &f.quote,
                &format!("[[tools]] {} {}", t.id, f.tool),
            )?;
            let micros = u64::try_from(decimal(&f.usd, 6)?).map_err(|_| "fee too large")?;
            fees.push((camel(&f.tool), micros));
        }
        fees.sort();
        tools.push((format!("{}_TOOLS", const_name(&t.id)), fees));
    }

    let mut cards = Vec::new();
    for c in &spec.cards {
        if let Some(t) = &c.tools
            && !spec.tools.iter().any(|x| &x.id == t)
        {
            return Err(format!("[[card]] {}: no [[tools]] {t}", c.name));
        }
        cards.push((
            c.name.clone(),
            cx.card(c).map_err(|e| format!("{}: {e}", c.name))?,
        ));
    }
    for c in &spec.conflicts {
        if !spec.cards.iter().any(|x| x.name == c.card) {
            return Err(format!("[[conflict]] for unknown card {}", c.card));
        }
    }
    for x in &spec.superseded {
        if !spec.cards.iter().any(|c| c.name == x.card) {
            return Err(format!("[[superseded]] for unknown card {}", x.card));
        }
    }
    for o in &spec.overrides {
        if !spec.cards.iter().any(|x| x.name == o.card) {
            return Err(format!("[[override]] for unknown card {}", o.card));
        }
    }

    let mut openrouter = Vec::new();
    let mut or_fees = BTreeMap::new();
    for slug in spec.openrouter_slugs() {
        let eps = cx.openrouter(&slug)?;
        for (_, fee) in &eps {
            if let Some(micros) = fee {
                or_fees.insert(or_search_name(*micros), *micros);
            }
        }
        openrouter.push((slug, eps.into_iter().map(|(e, _)| e).collect()));
    }
    let mut or_tools: Vec<(String, Vec<(String, u64)>)> = or_fees
        .into_iter()
        .map(|(name, micros)| (name, vec![("WebSearch".to_owned(), micros)]))
        .collect();
    // OR_SEARCH_10 before OR_SEARCH_5: the dearer first, as named.
    or_tools.sort_by(|a, b| b.1[0].1.cmp(&a.1[0].1));
    tools.extend(or_tools);

    if spec.pricing.len() != MODEL_ROUTES.len() {
        return Err(format!(
            "[[pricing]]: {} entries, the catalog has {} rows",
            spec.pricing.len(),
            MODEL_ROUTES.len()
        ));
    }
    let mut rows = Vec::new();
    for (p, route) in spec.pricing.iter().zip(MODEL_ROUTES) {
        if p.model != route.model {
            return Err(format!(
                "[[pricing]] {} is where the catalog has {}: keep catalog order",
                p.model, route.model
            ));
        }
        let list = if p.list_card == "list" {
            let lp = route.price;
            List::Inline(Tr {
                input: Rate::per_million(lp.input)?,
                output: Rate::per_million(lp.output)?,
                cache_read: Rate::per_million(lp.cache_read)?,
                cache_write_5m: Rate::per_million(lp.cache_write)?,
                cache_write_1h: None,
            })
        } else {
            if !spec.cards.iter().any(|c| c.name == p.list_card) {
                return Err(format!("{}: no [[card]] {}", p.model, p.list_card));
            }
            List::Card(p.list_card.clone())
        };
        let mut seen = Vec::new();
        let mut cost = Vec::new();
        for c in route.candidates.iter().chain(route.responses) {
            if seen.contains(&c.provider) {
                continue;
            }
            seen.push(c.provider);
            let pname = providers::by_id(c.provider).name;
            let want = p
                .cost
                .get(pname)
                .ok_or_else(|| format!("{}: [[pricing]] has no cost for {pname}", p.model))?;
            let cc = if want == "list" {
                Cost::List
            } else if let Some(slug) = want.strip_prefix("openrouter:") {
                Cost::OpenRouter(slug.to_owned())
            } else if let Some(r) = want.strip_prefix("unverified: ") {
                Cost::Unverified(r.to_owned())
            } else {
                if !spec.cards.iter().any(|c| &c.name == want) {
                    return Err(format!("{}: no [[card]] {want}", p.model));
                }
                Cost::Own(want.clone())
            };
            cost.push((format!("{:?}", c.provider), cc));
        }
        if cost.len() != p.cost.len() {
            return Err(format!(
                "{}: [[pricing]] names {} providers, the catalog row has {}",
                p.model,
                p.cost.len(),
                cost.len()
            ));
        }
        rows.push(Row {
            model: p.model.clone(),
            list,
            cost,
        });
    }

    // Every card and endpoint list is used.
    for (name, _) in &cards {
        let used = spec
            .pricing
            .iter()
            .any(|p| &p.list_card == name || p.cost.values().any(|v| v == name));
        if !used {
            return Err(format!("[[card]] {name}: no row uses it"));
        }
    }
    Ok(Table {
        credit_fee_bps,
        tools,
        cards,
        openrouter,
        rows,
    })
}
