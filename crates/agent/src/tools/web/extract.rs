//! Structured extraction for the `web` tool — `outline`/`locate`/`extract`/`table`.
//!
//! The point (ax's thesis) is to keep raw HTML *out of the model's context*: instead of dumping a page
//! and asking the model to parse it, the model names what it wants and gets back a compact,
//! token-cheap table. Output is header-once TSV-style rows with a trailing "N rows (M empty)" note —
//! ax's "never silent" principle, rendered as text since a tool has no stderr the model sees.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

use agent_core::ToolError;
use scraper::{ElementRef, Html, Selector};
use serde_json::Value;

use super::Mode;
use super::whereexpr::Filter;

/// Rows past this are dropped with a note — ax's own cap. Keeps a big listing from flooding context.
const DEFAULT_MAX_ITEMS: usize = 50;

pub fn run(mode: Mode, input: &Value, html: &str) -> Result<String, ToolError> {
    let doc = Html::parse_document(html);
    match mode {
        Mode::Outline => Ok(outline(&doc)),
        Mode::Locate => locate(&doc, input),
        Mode::Extract => extract(&doc, input),
        Mode::Table => table(&doc, input),
        // fetch/markdown never reach here.
        Mode::Fetch | Mode::Markdown => unreachable!(),
    }
}

fn max_items(input: &Value) -> usize {
    input
        .get("max_items")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_MAX_ITEMS)
}

fn parse_selector(sel: &str) -> Result<Selector, ToolError> {
    Selector::parse(sel)
        .map_err(|e| ToolError::InvalidInput(format!("invalid CSS selector `{sel}`: {e}")))
}

/// The visible text of an element, whitespace-collapsed to one line.
fn text_of(el: &ElementRef) -> String {
    // Fold every text fragment through one whitespace-collapsing writer — a single output
    // allocation, no intermediate concatenation.
    let mut out = String::new();
    let mut pending = false;
    for frag in el.text() {
        push_collapsed(&mut out, &mut pending, frag);
    }
    out
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::new();
    let mut pending = false;
    push_collapsed(&mut out, &mut pending, s);
    out
}

/// Append `s`'s characters to `out`, collapsing internal whitespace runs to a single space and
/// trimming leading/trailing whitespace. `pending` carries "saw whitespace, awaiting the next token
/// character" across calls, so folding several fragments collapses exactly as if they had been
/// concatenated first (a boundary with no whitespace joins into one token; a whitespace-only
/// fragment between two tokens still yields a single space).
fn push_collapsed(out: &mut String, pending: &mut bool, s: &str) {
    for ch in s.chars() {
        if ch.is_whitespace() {
            // Only arm a space once we've emitted a token — leading whitespace is dropped, and a
            // deferred space never lands at the end because it's flushed only before a token char.
            if !out.is_empty() {
                *pending = true;
            }
        } else {
            if *pending {
                out.push(' ');
                *pending = false;
            }
            out.push(ch);
        }
    }
}

// ---- outline -------------------------------------------------------------------------------------

/// The page's repeating structure: `tag.class` shapes and how many of each, most common first. Gives
/// the model a map of what's on the page without a byte of raw HTML.
fn outline(doc: &Html) -> String {
    let all = match Selector::parse("*") {
        Ok(s) => s,
        Err(_) => return "outline: could not parse the document".to_string(),
    };
    let mut counts: HashMap<String, usize> = HashMap::new();
    for el in doc.select(&all) {
        let v = el.value();
        let mut key = v.name().to_string();
        // Include up to two classes — enough to distinguish `div.card` from `div.footer` without the
        // long utility-class soup some pages carry.
        for c in v.classes().take(2) {
            key.push('.');
            key.push_str(c);
        }
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut rows: Vec<(String, usize)> = counts.into_iter().collect();
    // Most frequent first, then alphabetical — repeating structures (a list of cards) float to the top.
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut out = String::from("count\tselector\n");
    for (sel, n) in rows.iter().take(60) {
        out.push_str(&format!("{n}\t{sel}\n"));
    }
    out.push_str(&format!("\n[{} distinct shapes]", rows.len()));
    out
}

// ---- locate --------------------------------------------------------------------------------------

/// Which selector(s) hold a given text — the "I can see this string on the page, how do I select it?"
/// question, answered without the model reading HTML. Reports the tightest `tag.class` for each element
/// whose text contains the target.
fn locate(doc: &Html, input: &Value) -> Result<String, ToolError> {
    let needle = input
        .get("text")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ToolError::InvalidInput("locate mode needs `text` to find".into()))?;
    let needle_lc = needle.to_lowercase();

    let all = parse_selector("*")?;
    // Count selector frequencies among matching *leaf-ish* elements (those whose own text — not a
    // child's — contains the needle), so we point at the element that actually holds the string.
    let mut hits: BTreeMap<String, usize> = BTreeMap::new();
    for el in doc.select(&all) {
        // Direct text of this element only (not descendants), collapsed.
        let direct: String = el
            .children()
            .filter_map(|c| c.value().as_text().map(|t| &**t))
            .collect();
        if collapse_ws(&direct).to_lowercase().contains(&needle_lc) {
            let v = el.value();
            let mut key = v.name().to_string();
            if let Some(c) = v.classes().next() {
                key.push('.');
                key.push_str(c);
            }
            *hits.entry(key).or_insert(0) += 1;
        }
    }
    if hits.is_empty() {
        return Ok(format!("no element's own text contains {needle:?}"));
    }
    let mut rows: Vec<(String, usize)> = hits.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut out = format!("selectors whose text contains {needle:?}:\ncount\tselector\n");
    for (sel, n) in &rows {
        out.push_str(&format!("{n}\t{sel}\n"));
    }
    Ok(out)
}

// ---- extract (the --row DSL) ---------------------------------------------------------------------

/// One output column: a name and how to pull it from a row element.
struct Field {
    name: String,
    selector: Selector,
    /// `Some(attr)` reads that attribute; `None` reads text.
    attr: Option<String>,
}

/// Parse the compact `--row` form `"title=a, href=a@href"`, or the structured object form. `a@href`
/// reads the `href` attribute of the first `a`; a bare selector reads text.
fn parse_fields(input: &Value) -> Result<Vec<Field>, ToolError> {
    let raw = input
        .get("fields")
        .ok_or_else(|| ToolError::InvalidInput("extract mode needs `fields`".into()))?;

    let pairs: Vec<(String, String)> = match raw {
        Value::String(s) => s
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| {
                let (name, sel) = p.split_once('=').ok_or_else(|| {
                    ToolError::InvalidInput(format!(
                        "bad field `{p}` — expected name=selector (e.g. `title=h2` or `href=a@href`)"
                    ))
                })?;
                Ok((name.trim().to_string(), sel.trim().to_string()))
            })
            .collect::<Result<_, ToolError>>()?,
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| {
                let sel = v
                    .as_str()
                    .ok_or_else(|| ToolError::InvalidInput(format!("field `{k}` must be a selector string")))?;
                Ok((k.clone(), sel.to_string()))
            })
            .collect::<Result<_, ToolError>>()?,
        _ => return Err(ToolError::InvalidInput("`fields` must be a string or object".into())),
    };
    if pairs.is_empty() {
        return Err(ToolError::InvalidInput("`fields` names no columns".into()));
    }

    pairs
        .into_iter()
        .map(|(name, spec)| {
            let (sel, attr) = match spec.split_once('@') {
                Some((s, a)) => (s.trim(), Some(a.trim().to_string())),
                None => (spec.as_str(), None),
            };
            Ok(Field {
                name,
                selector: parse_selector(sel)?,
                attr,
            })
        })
        .collect()
}

fn extract(doc: &Html, input: &Value) -> Result<String, ToolError> {
    let row_sel = input
        .get("selector")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            ToolError::InvalidInput("extract mode needs a `selector` for each row".into())
        })?;
    let row_selector = parse_selector(row_sel)?;
    let fields = parse_fields(input)?;

    let filter = match input
        .get("where")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        Some(expr) => Some(Filter::parse(expr).map_err(ToolError::InvalidInput)?),
        None => None,
    };
    let cap = max_items(input);

    let mut rows: Vec<HashMap<String, String>> = Vec::new();
    let mut empty_fields = 0usize;
    let mut total_matched = 0usize;
    for row_el in doc.select(&row_selector) {
        let mut row = HashMap::new();
        for field in &fields {
            let value = row_el
                .select(&field.selector)
                .next()
                .map(|el| match &field.attr {
                    Some(attr) => el
                        .value()
                        .attr(attr)
                        .map(str::to_string)
                        .unwrap_or_default(),
                    None => text_of(&el),
                })
                .unwrap_or_default();
            if value.is_empty() {
                empty_fields += 1;
            }
            row.insert(field.name.clone(), value);
        }
        if filter.as_ref().is_none_or(|f| f.matches(&row)) {
            total_matched += 1;
            if rows.len() < cap {
                rows.push(row);
            }
        }
    }

    Ok(render_rows(
        &fields.iter().map(|f| f.name.clone()).collect::<Vec<_>>(),
        &rows,
        total_matched,
        empty_fields,
        cap,
    ))
}

// ---- table ---------------------------------------------------------------------------------------

/// Convert HTML `<table>`s to header-keyed rows. `selector` narrows to a specific table; absent, the
/// first table on the page is used.
fn table(doc: &Html, input: &Value) -> Result<String, ToolError> {
    let sel = input
        .get("selector")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("table");
    let table_sel = parse_selector(sel)?;
    let Some(table) = doc.select(&table_sel).next() else {
        return Ok(format!("no table matched `{sel}`"));
    };

    let tr = parse_selector("tr")?;
    let th = parse_selector("th")?;
    let td = parse_selector("td")?;

    let mut header: Vec<String> = Vec::new();
    let mut rows: Vec<HashMap<String, String>> = Vec::new();
    let cap = max_items(input);
    let mut total = 0usize;

    for row_el in table.select(&tr) {
        let ths: Vec<String> = row_el.select(&th).map(|c| text_of(&c)).collect();
        if header.is_empty() && !ths.is_empty() {
            header = ths;
            continue;
        }
        let cells: Vec<String> = row_el.select(&td).map(|c| text_of(&c)).collect();
        if cells.is_empty() {
            continue;
        }
        // Fill a header if the table had none (`col1`, `col2`, …).
        if header.is_empty() {
            header = (1..=cells.len()).map(|i| format!("col{i}")).collect();
        }
        total += 1;
        if rows.len() < cap {
            let row = header
                .iter()
                .cloned()
                .zip(cells.into_iter().chain(std::iter::repeat(String::new())))
                .collect();
            rows.push(row);
        }
    }
    if header.is_empty() {
        return Ok(format!("`{sel}` matched a table with no rows"));
    }
    Ok(render_rows(&header, &rows, total, 0, cap))
}

// ---- shared rendering ----------------------------------------------------------------------------

/// Header-once TSV, then a trailing note. Tabs/newlines in a value are spaced out so a row stays one
/// line — the whole point is a compact, greppable table the model reads cheaply.
fn render_rows(
    columns: &[String],
    rows: &[HashMap<String, String>],
    total_matched: usize,
    empty_fields: usize,
    cap: usize,
) -> String {
    let mut out = String::new();
    out.push_str(&columns.join("\t"));
    out.push('\n');
    for row in rows {
        for (i, c) in columns.iter().enumerate() {
            if i > 0 {
                out.push('\t');
            }
            out.push_str(&clean_cell(row.get(c).map(String::as_str).unwrap_or("")));
        }
        out.push('\n');
    }

    let shown = rows.len();
    out.push('\n');
    out.push_str(&format!("[{shown} row(s)"));
    if total_matched > shown {
        out.push_str(&format!(
            "; {} more not shown (max_items={cap})",
            total_matched - shown
        ));
    }
    if empty_fields > 0 {
        out.push_str(&format!("; {empty_fields} empty field(s)"));
    }
    out.push(']');
    out
}

fn clean_cell(s: &str) -> Cow<'_, str> {
    if s.contains(['\t', '\n', '\r']) {
        Cow::Owned(s.replace(['\t', '\n', '\r'], " "))
    } else {
        Cow::Borrowed(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const LISTINGS: &str = r#"<html><body>
        <div class="card"><h2>Alpha</h2><a href="/a">buy</a><span class="price">150</span></div>
        <div class="card"><h2>Beta</h2><a href="/b">buy</a><span class="price">50</span></div>
        <div class="card"><h2>Gamma</h2><a href="/g">buy</a><span class="price">999</span></div>
    </body></html>"#;

    fn go(mode: Mode, input: Value, html: &str) -> String {
        run(mode, &input, html).unwrap()
    }

    #[test]
    fn extract_pulls_keyed_rows_with_text_and_attributes() {
        let out = go(
            Mode::Extract,
            json!({ "selector": ".card", "fields": "title=h2, href=a@href, price=.price" }),
            LISTINGS,
        );
        assert!(out.contains("title\thref\tprice"), "{out}");
        assert!(out.contains("Alpha\t/a\t150"), "{out}");
        assert!(out.contains("Gamma\t/g\t999"), "{out}");
        assert!(out.contains("[3 row(s)]"), "{out}");
    }

    #[test]
    fn extract_applies_a_where_filter() {
        let out = go(
            Mode::Extract,
            json!({ "selector": ".card", "fields": "title=h2, price=.price", "where": "price > 100" }),
            LISTINGS,
        );
        assert!(out.contains("Alpha"), "{out}");
        assert!(out.contains("Gamma"), "{out}");
        assert!(
            !out.contains("Beta"),
            "Beta (50) must be filtered out: {out}"
        );
    }

    #[test]
    fn extract_honors_max_items() {
        let out = go(
            Mode::Extract,
            json!({ "selector": ".card", "fields": "title=h2", "max_items": 2 }),
            LISTINGS,
        );
        assert!(out.contains("[2 row(s); 1 more not shown"), "{out}");
    }

    #[test]
    fn extract_accepts_the_structured_fields_form() {
        let out = go(
            Mode::Extract,
            json!({ "selector": ".card", "fields": { "t": "h2", "u": "a@href" } }),
            LISTINGS,
        );
        assert!(out.contains("Alpha"), "{out}");
        assert!(out.contains("/a"), "{out}");
    }

    #[test]
    fn table_mode_reads_header_keyed_rows() {
        let html = r#"<table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>Ada</td><td>36</td></tr>
            <tr><td>Alan</td><td>41</td></tr>
        </table>"#;
        let out = go(Mode::Table, json!({}), html);
        assert!(out.contains("Name\tAge"), "{out}");
        assert!(out.contains("Ada\t36"), "{out}");
        assert!(out.contains("Alan\t41"), "{out}");
    }

    #[test]
    fn outline_reports_repeating_structure() {
        let out = go(Mode::Outline, json!({}), LISTINGS);
        assert!(
            out.contains("div.card"),
            "the repeating card must surface: {out}"
        );
        assert!(out.contains("count\tselector"), "{out}");
    }

    #[test]
    fn locate_finds_the_selector_holding_text() {
        let out = go(Mode::Locate, json!({ "text": "Beta" }), LISTINGS);
        assert!(out.contains("h2"), "Beta lives in an h2: {out}");
    }

    #[test]
    fn a_bad_selector_is_invalid_input() {
        let err = run(
            Mode::Extract,
            &json!({ "selector": ">>bad", "fields": "t=h2" }),
            LISTINGS,
        )
        .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn extract_without_fields_is_rejected() {
        let err = run(Mode::Extract, &json!({ "selector": ".card" }), LISTINGS).unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }
}
