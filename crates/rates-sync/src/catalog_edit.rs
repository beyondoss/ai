//! The text of a catalog-drift edit: rendering a row, a candidate list and the truth file's
//! entries, and splicing them into `crates/providers/src/catalog.rs` and
//! `verify/catalog_truth.toml`. Pure string to string: the same edit on the same files always
//! gives the same bytes.

use crate::Result;
use providers::catalog::{FEATURE_NAMES, INPUT_NAMES, ModelCard, UNPUBLISHED_MAX_OUTPUT};
use providers::{ProviderId, WireFormat};
use std::fmt::Write as _;

/// One candidate of a generated row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewCand {
    pub provider: ProviderId,
    pub id: String,
    pub path: &'static str,
}

#[derive(Clone, Debug)]
pub enum Edit {
    Add(Addition),
    Retire(Removal),
}

/// A successor row and its truth-file entries.
#[derive(Clone, Debug)]
pub struct Addition {
    pub model: String,
    /// The catalog row it goes after (rows are sorted by name); `None`: first.
    pub after: Option<String>,
    /// The `ModelRoute { … },` text.
    pub row: String,
    /// (the predecessor whose `[[row]]` it follows, the `[[row]]` text).
    pub truth_row: Option<(String, String)>,
    /// (the predecessor card it follows, the `[[card]]` text).
    pub cards: Vec<(String, String)>,
    /// The `[[pricing]]` text, placed in catalog order.
    pub pricing: String,
    /// Snapshot ids the new cards read, to take from the run's fresh snapshots.
    pub refresh: Vec<String>,
    pub new_slugs: Vec<String>,
}

/// Candidates leaving a row, or the row leaving the catalog for an alias.
#[derive(Clone, Debug)]
pub struct Removal {
    pub model: String,
    /// The row's new `candidates:` value; `None` when the row goes.
    pub candidates: Option<String>,
    /// The row's new `responses:` value, when it changes.
    pub responses: Option<String>,
    /// The row goes, and its name resolves to this row.
    pub alias_to: Option<String>,
    /// The row's new `[[pricing]]` `cost = …` line; `None` when the row goes.
    pub cost: Option<String>,
    /// `[[card]]`s no row uses any more.
    pub drop_cards: Vec<String>,
    /// `[[retired]]` blocks to add.
    pub retired: Vec<String>,
}

/// What a generated row is made of.
pub struct NewRow<'a> {
    pub model: &'a str,
    pub wire: WireFormat,
    pub candidates: &'a [NewCand],
    pub responses: &'a [NewCand],
    /// input, output, cache read, cache write.
    pub price: &'a [String; 4],
    /// The predecessor's card: limits and capability bits carry over.
    pub card: ModelCard,
    pub name: &'a str,
    pub created: u64,
}

/// `1000000` → `1_000_000`.
fn grouped(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push('_');
        }
        out.push(c);
    }
    out
}

fn bits(v: u8, table: &[(u8, &str)], prefix: &str) -> String {
    let names: Vec<String> = table
        .iter()
        .filter(|(b, _)| v & b != 0)
        .map(|(_, n)| format!("{prefix}{}", n.to_ascii_uppercase()))
        .collect();
    if names.is_empty() {
        "0".into()
    } else {
        names.join(" | ")
    }
}

/// A candidate list as the value of a `ModelRoute` field, rustfmt's layout at that depth.
pub fn candidates(c: &[NewCand]) -> String {
    let one = |c: &NewCand, pad: &str| {
        format!(
            "Candidate {{\n{pad}    provider: ProviderId::{:?},\n{pad}    upstream_model: \"{}\",\n{pad}    path: \"{}\",\n{pad}}}",
            c.provider, c.id, c.path
        )
    };
    match c {
        [] => "&[]".into(),
        [c] => format!("&[{}]", one(c, "        ")),
        _ => {
            let mut s = String::from("&[\n");
            for c in c {
                let _ = writeln!(s, "            {},", one(c, "            "));
            }
            s.push_str("        ]");
            s
        }
    }
}

/// A whole `ModelRoute { … },` entry, indented for `MODEL_ROUTES`.
pub fn row(r: &NewRow<'_>) -> String {
    let c = r.card;
    let [i, o, cr, cw] = r.price;
    let wire = match r.wire {
        WireFormat::Anthropic => "Anthropic",
        _ => "OpenAi",
    };
    let card = if c.max_output_published || c.max_output_tokens != UNPUBLISHED_MAX_OUTPUT {
        format!(
            "card(\n            \"{}\",\n            \"{}\",\n            {},\n            {},\n            {},\n            {},\n            {},\n        )",
            r.name,
            c.owned_by,
            r.created,
            grouped(c.context_window),
            grouped(c.max_output_tokens),
            bits(c.input, &INPUT_NAMES, "IN_"),
            bits(c.features, &FEATURE_NAMES, ""),
        )
    } else {
        format!(
            "card_unpublished_output(\n            \"{}\",\n            \"{}\",\n            {},\n            {},\n            {},\n            {},\n        )",
            r.name,
            c.owned_by,
            r.created,
            grouped(c.context_window),
            bits(c.input, &INPUT_NAMES, "IN_"),
            bits(c.features, &FEATURE_NAMES, ""),
        )
    };
    format!(
        "    ModelRoute {{\n        model: \"{}\",\n        wire: WireFormat::{wire},\n        candidates: {},\n        responses: {},\n        price: price(\"{i}\", \"{o}\", \"{cr}\", \"{cw}\"),\n        card: {card},\n    }},\n",
        r.model,
        candidates(r.candidates),
        candidates(r.responses),
    )
}

/// A `[[card]]` block.
pub fn card_block(c: &crate::spec::CardSpec) -> String {
    let mut s = format!("[[card]]\nname = \"{}\"\nrow = \"{}\"\n", c.name, c.row);
    for (k, v) in [
        ("geo_us", &c.geo_us),
        ("tools", &c.tools),
        ("unpublished_output", &c.unpublished_output),
    ] {
        if let Some(v) = v {
            let _ = writeln!(s, "{k} = \"{v}\"");
        }
    }
    s
}

/// A `[[row]]` block for a generated row: the list price as the list card's page gives it, the
/// release time, and the predecessor's recorded capability bits (the feature class is the same).
pub fn truth_row(
    model: &str,
    source: &str,
    date: &str,
    price: &[String; 4],
    created: u64,
    pred: &toml::Value,
) -> String {
    let [i, o, cr, cw] = price;
    let mut s = format!(
        "[[row]]\nmodel = \"{model}\"\nsource = [\"{source}\"]\ndate = \"{date}\"\ninput = \"{i}\"\noutput = \"{o}\"\ncache_read = \"{cr}\"\ncache_write = \"{cw}\"\ncreated = {created}\n"
    );
    if let Some(v) = pred.get("owned_by").and_then(toml::Value::as_str) {
        let _ = writeln!(s, "owned_by = \"{v}\"");
    }
    for k in [
        "input_present",
        "input_absent",
        "features_present",
        "features_absent",
    ] {
        if let Some(a) = pred.get(k).and_then(toml::Value::as_array) {
            let names: Vec<String> = a
                .iter()
                .filter_map(|x| x.as_str().map(|x| format!("\"{x}\"")))
                .collect();
            let _ = writeln!(s, "{k} = [{}]", names.join(", "));
        }
    }
    s
}

/// `cost = { anthropic = "list", openrouter = "openrouter:…" }`.
pub fn cost_line(cost: &[(String, String)]) -> String {
    let parts: Vec<String> = cost.iter().map(|(h, c)| format!("{h} = \"{c}\"")).collect();
    format!("cost = {{ {} }}", parts.join(", "))
}

/// A `[[pricing]]` block.
pub fn pricing_block(model: &str, list_card: &str, cost: &[(String, String)]) -> String {
    format!(
        "[[pricing]]\nmodel = \"{model}\"\nlist_card = \"{list_card}\"\n{}\n",
        cost_line(cost)
    )
}

/// A `[[retired]]` block: `retired` when the date has come, else `retires`.
pub fn retired_block(
    model: Option<&str>,
    provider: &str,
    id: &str,
    source: &str,
    date: &str,
    today: &str,
    note: &str,
) -> String {
    let mut s = String::from("[[retired]]\n");
    if let Some(m) = model {
        let _ = writeln!(s, "model = \"{m}\"");
    }
    let key = if date <= today { "retired" } else { "retires" };
    let _ = write!(
        s,
        "provider = \"{provider}\"\nid = \"{id}\"\nsource = \"{source}\"\n{key} = \"{date}\"\nnote = \"{}\"\n",
        note.replace('\\', "\\\\").replace('"', "\\\"")
    );
    s
}

// ---------------------------------------------------------------------------------------------
// Splicing
// ---------------------------------------------------------------------------------------------

fn one_at(hay: &str, needle: &str, what: &str) -> Result<usize> {
    let mut it = hay.match_indices(needle);
    match (it.next(), it.next()) {
        (Some((i, _)), None) => Ok(i),
        (None, _) => Err(format!("{what}: not found")),
        _ => Err(format!("{what}: found more than once")),
    }
}

/// The byte range of a `MODEL_ROUTES` entry, from its `    ModelRoute {` line through `    },\n`.
fn row_range(text: &str, model: &str) -> Result<(usize, usize)> {
    let at = one_at(text, &format!("\n        model: \"{model}\",\n"), model)?;
    let start = text[..=at]
        .rfind("\n    ModelRoute {\n")
        .ok_or_else(|| format!("{model}: no `ModelRoute {{` before it"))?
        + 1;
    let end = at
        + text[at..]
            .find("\n    },\n")
            .ok_or_else(|| format!("{model}: unterminated"))?
        + "\n    },\n".len();
    Ok((start, end))
}

/// Replace a field's value (`        {field}: … ,` up to the next field) within a row.
fn replace_field(row: &str, field: &str, next: &str, value: &str) -> Result<String> {
    let start = one_at(row, &format!("\n        {field}: "), field)? + 1;
    let end = one_at(row, &format!("\n        {next}: "), next)? + 1;
    Ok(format!(
        "{}        {field}: {value},\n{}",
        &row[..start],
        &row[end..]
    ))
}

pub fn apply_catalog(text: &str, edit: &Edit) -> Result<String> {
    match edit {
        Edit::Add(a) => {
            let at = match &a.after {
                Some(m) => row_range(text, m)?.1,
                None => {
                    let head = one_at(
                        text,
                        "pub const MODEL_ROUTES: &[ModelRoute] = &[\n",
                        "MODEL_ROUTES",
                    )?;
                    head + text[head..]
                        .find("\n    ModelRoute {\n")
                        .ok_or("MODEL_ROUTES: no row")?
                        + 1
                }
            };
            Ok(format!("{}{}{}", &text[..at], a.row, &text[at..]))
        }
        Edit::Retire(r) => {
            let (start, end) = row_range(text, &r.model)?;
            let mut out = text.to_owned();
            if let Some(to) = &r.alias_to {
                out.replace_range(start..end, "");
                out = add_alias(&out, &r.model, to)?;
            } else {
                let mut row = text[start..end].to_owned();
                if let Some(c) = &r.candidates {
                    row = replace_field(&row, "candidates", "responses", c)?;
                }
                if let Some(c) = &r.responses {
                    row = replace_field(&row, "responses", "price", c)?;
                }
                out.replace_range(start..end, &row);
            }
            Ok(out)
        }
    }
}

/// Add `(old, new)` to `ALIASES`, kept sorted.
fn add_alias(text: &str, old: &str, new: &str) -> Result<String> {
    let head = "pub const ALIASES: &[(&str, &str)] = &[";
    let start = one_at(text, head, "ALIASES")? + head.len();
    let end = start + text[start..].find("];").ok_or("ALIASES: unterminated")?;
    let mut pairs: Vec<(String, String)> = Vec::new();
    for part in text[start..end].split('(').skip(1) {
        let q: Vec<&str> = part.split('"').collect();
        if q.len() >= 4 {
            pairs.push((q[1].to_owned(), q[3].to_owned()));
        }
    }
    pairs.retain(|(a, _)| a != old);
    pairs.push((old.to_owned(), new.to_owned()));
    pairs.sort();
    let mut body = String::from("\n");
    for (a, b) in &pairs {
        let _ = writeln!(body, "    (\"{a}\", \"{b}\"),");
    }
    Ok(format!("{}{body}{}", &text[..start], &text[end..]))
}

/// The byte range of the truth file's block `[[{kind}]]` whose `{key} = "{value}"` line follows
/// its header, through the blank line after it.
fn block_range(text: &str, kind: &str, key: &str, value: &str) -> Result<(usize, usize)> {
    let start = one_at(
        text,
        &format!("[[{kind}]]\n{key} = \"{value}\"\n"),
        &format!("[[{kind}]] {value}"),
    )?;
    if start > 0 && !text[..start].ends_with('\n') {
        return Err(format!("[[{kind}]] {value}: not at a line start"));
    }
    let end = text[start..]
        .find("\n\n")
        .map_or(text.len(), |i| start + i + 2);
    Ok((start, end))
}

fn insert_after_block(
    text: &str,
    kind: &str,
    key: &str,
    after: &str,
    block: &str,
) -> Result<String> {
    let (_, end) = block_range(text, kind, key, after)?;
    let sep = if text[..end].ends_with("\n\n") {
        ""
    } else {
        "\n"
    };
    Ok(format!("{}{sep}{block}\n{}", &text[..end], &text[end..]))
}

pub fn apply_truth(text: &str, edit: &Edit) -> Result<String> {
    match edit {
        Edit::Add(a) => {
            let mut t = text.to_owned();
            if let Some((pred, block)) = &a.truth_row {
                t = insert_after_block(&t, "row", "model", pred, block)?;
            }
            for (pred, block) in &a.cards {
                t = insert_after_block(&t, "card", "name", pred, block)?;
            }
            t = match &a.after {
                Some(m) => insert_after_block(&t, "pricing", "model", m, &a.pricing)?,
                None => {
                    let at = one_at(&t, "[[pricing]]\n", "[[pricing]]").or_else(|_| {
                        t.find("\n[[pricing]]\n")
                            .map(|i| i + 1)
                            .ok_or_else(|| "[[pricing]]: none".to_owned())
                    })?;
                    format!("{}{}\n{}", &t[..at], a.pricing, &t[at..])
                }
            };
            Ok(t)
        }
        Edit::Retire(r) => {
            let mut t = text.to_owned();
            match &r.cost {
                Some(line) => {
                    let (s, e) = block_range(&t, "pricing", "model", &r.model)?;
                    let block = &t[s..e];
                    let cs = one_at(block, "\ncost = {", "cost")? + 1;
                    let ce = cs + block[cs..].find('\n').ok_or("cost: unterminated")?;
                    let new = format!("{}{line}{}", &block[..cs], &block[ce..]);
                    t.replace_range(s..e, &new);
                }
                None => {
                    for kind in ["pricing", "row"] {
                        if let Ok((s, e)) = block_range(&t, kind, "model", &r.model) {
                            t.replace_range(s..e, "");
                        }
                    }
                }
            }
            for c in &r.drop_cards {
                let (s, e) = block_range(&t, "card", "name", c)?;
                t.replace_range(s..e, "");
            }
            if !r.retired.is_empty() {
                // After the last [[retired]] block.
                let last = t.rfind("\n[[retired]]\n").ok_or("[[retired]]: none")? + 1;
                let end = t[last..].find("\n\n").map_or(t.len(), |i| last + i + 2);
                let mut add = String::new();
                for b in &r.retired {
                    add.push_str(b);
                    add.push('\n');
                }
                t.insert_str(end, &add);
            }
            Ok(t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_grouped_as_rustfmt_leaves_them() {
        assert_eq!(grouped(200_000), "200_000");
        assert_eq!(grouped(1_048_575), "1_048_575");
        assert_eq!(grouped(64), "64");
    }

    #[test]
    fn aliases_stay_sorted() {
        let t = "pub const ALIASES: &[(&str, &str)] = &[];\n";
        let t = add_alias(t, "b-1", "b-2").unwrap_or_default();
        let t = add_alias(&t, "a-1", "a-2").unwrap_or_default();
        assert_eq!(
            t,
            "pub const ALIASES: &[(&str, &str)] = &[\n    (\"a-1\", \"a-2\"),\n    (\"b-1\", \"b-2\"),\n];\n"
        );
    }
}
