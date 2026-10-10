//! Strict readers for the shapes the vendors publish prices in: GitHub-flavoured markdown tables,
//! HTML tables, and dollar amounts. Anything that does not have the expected shape is an error,
//! never a guess.

use crate::Result;
use crate::dec::Rate;

/// One markdown table.
#[derive(Debug, Clone)]
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// The non-blank, non-table lines before the table, nearest first (at most four).
    pub context: Vec<String>,
    /// The 1-based line of the header.
    pub line: usize,
}

impl Table {
    /// Fail unless the header is exactly `want`.
    pub fn expect_header(&self, want: &[&str], what: &str) -> Result<()> {
        if self
            .header
            .iter()
            .map(String::as_str)
            .ne(want.iter().copied())
        {
            return Err(format!(
                "{what} (line {}): header changed: {:?}, expected {:?}",
                self.line, self.header, want
            ));
        }
        Ok(())
    }
}

fn split_row(line: &str) -> Option<Vec<String>> {
    let t = line.trim();
    if !t.starts_with('|') || !t.ends_with('|') || t.len() < 2 {
        return None;
    }
    let inner = &t[1..t.len() - 1];
    let mut cells = Vec::new();
    let mut cur = String::new();
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'|') => {
                cur.push('|');
                chars.next();
            }
            '|' => cells.push(std::mem::take(&mut cur).trim().to_owned()),
            _ => cur.push(c),
        }
    }
    cells.push(cur.trim().to_owned());
    Some(cells)
}

fn is_separator(cells: &[String]) -> bool {
    !cells.is_empty()
        && cells.iter().all(|c| {
            let c = c.trim_matches(':');
            !c.is_empty() && c.bytes().all(|b| b == b'-')
        })
}

/// Every table in a markdown document, in order.
pub fn tables(md: &str) -> Result<Vec<Table>> {
    let lines: Vec<&str> = md.lines().collect();
    let mut out = Vec::new();
    let mut context: Vec<String> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some(header) = split_row(line) {
            let sep = lines.get(i + 1).and_then(|l| split_row(l));
            match sep {
                Some(sep) if is_separator(&sep) => {
                    if sep.len() != header.len() {
                        return Err(format!(
                            "line {}: separator width differs from header",
                            i + 2
                        ));
                    }
                    let mut rows = Vec::new();
                    let mut j = i + 2;
                    while let Some(r) = lines.get(j).and_then(|l| split_row(l)) {
                        if r.len() != header.len() {
                            return Err(format!(
                                "line {}: {} cells, the header has {}",
                                j + 1,
                                r.len(),
                                header.len()
                            ));
                        }
                        rows.push(r);
                        j += 1;
                    }
                    out.push(Table {
                        header,
                        rows,
                        context: context.iter().rev().take(4).cloned().collect(),
                        line: i + 1,
                    });
                    context.clear();
                    i = j;
                    continue;
                }
                _ => {}
            }
        }
        let t = line.trim();
        if !t.is_empty() {
            context.push(t.to_owned());
        }
        i += 1;
    }
    Ok(out)
}

/// The one table whose context's nearest line is exactly `label`.
pub fn table_labelled<'a>(ts: &'a [Table], label: &str) -> Result<&'a Table> {
    let hits: Vec<&Table> = ts
        .iter()
        .filter(|t| t.context.first().map(String::as_str) == Some(label))
        .collect();
    match hits[..] {
        [t] => Ok(t),
        _ => Err(format!(
            "expected exactly one table under {label:?}, found {}",
            hits.len()
        )),
    }
}

/// Remove `<sup>…</sup>` footnote markers.
pub fn strip_sup(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("<sup>") {
        out.push_str(&rest[..i]);
        match rest[i..].find("</sup>") {
            Some(j) => rest = &rest[i + j + "</sup>".len()..],
            None => {
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

/// `$1.25`, `\$1.25`, `$1,000` → the rate (per million tokens: the caller's unit).
pub fn dollars(cell: &str) -> Result<Rate> {
    let c = cell.trim();
    let c = c.strip_prefix('\\').unwrap_or(c);
    let n = c
        .strip_prefix('$')
        .ok_or_else(|| format!("{cell:?} is not a dollar amount"))?;
    Rate::per_million(&n.replace(',', ""))
}

/// The cells of every `<tr>` of an HTML fragment, as plain text.
pub fn html_rows(html: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for tr in html.split("<tr").skip(1) {
        let tr = tr.split("</tr>").next().unwrap_or("");
        let mut cells = Vec::new();
        for (k, part) in tr.match_indices("<t") {
            let after = &tr[k + part.len()..];
            let cell = ["d>", "d ", "h>", "h "]
                .iter()
                .any(|p| after.starts_with(p));
            if !cell {
                continue;
            }
            let Some(gt) = after.find('>') else { continue };
            let body = &after[gt + 1..];
            let end = body
                .find("</td>")
                .into_iter()
                .chain(body.find("</th>"))
                .min()
                .unwrap_or(body.len());
            cells.push(html_text(&body[..end]));
        }
        rows.push(cells);
    }
    rows
}

/// Tags stripped, the common entities decoded, whitespace collapsed.
pub fn html_text(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => {
                in_tag = true;
                out.push(' ');
            }
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    let out = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace('\u{200b}', "");
    collapse(&out)
}

pub fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_table_and_its_label() {
        let md =
            "intro\n### Prices\n\n| A | B |\n| --- | :-- |\n| x | \\$1 |\n| y | 2 \\| 3 |\nafter\n";
        let ts = tables(md).unwrap();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].context[0], "### Prices");
        assert_eq!(ts[0].rows[1][1], "2 | 3");
        assert_eq!(
            dollars(&ts[0].rows[0][1]).unwrap().table_text().unwrap(),
            "1"
        );
        assert!(tables("| a | b |\n| - | - |\n| 1 |\n").is_err());
    }

    #[test]
    fn reads_html_rows() {
        let rows = html_rows("<table><tr><td>A&amp;B</td><td><p>$0.3</p></td></tr></table>");
        assert_eq!(rows, vec![vec!["A&B".to_owned(), "$0.3".to_owned()]]);
    }
}
