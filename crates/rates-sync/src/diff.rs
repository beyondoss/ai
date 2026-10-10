//! A line diff for the drift report: what changed in the generated table, each hunk headed by the
//! card or endpoint list it falls in.

/// Unified-style hunks between `old` and `new`, or `None` when they are equal.
pub fn lines(old: &str, new: &str) -> Option<String> {
    if old == new {
        return None;
    }
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let pre = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suf = a[pre..]
        .iter()
        .rev()
        .zip(b[pre..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let (am, bm) = (&a[pre..a.len() - suf], &b[pre..b.len() - suf]);
    // LCS over the changed middle; it is small for a real price change.
    let (n, m) = (am.len(), bm.len());
    let mut ops: Vec<(char, &str)> = Vec::new();
    if n.saturating_mul(m) <= 4_000_000 {
        let mut dp = vec![0u32; (n + 1) * (m + 1)];
        let at = |i: usize, j: usize| i * (m + 1) + j;
        for i in (0..n).rev() {
            for j in (0..m).rev() {
                dp[at(i, j)] = if am[i] == bm[j] {
                    dp[at(i + 1, j + 1)] + 1
                } else {
                    dp[at(i + 1, j)].max(dp[at(i, j + 1)])
                };
            }
        }
        let (mut i, mut j) = (0, 0);
        while i < n || j < m {
            if i < n && j < m && am[i] == bm[j] {
                ops.push((' ', am[i]));
                i += 1;
                j += 1;
            } else if j < m && (i == n || dp[at(i, j + 1)] > dp[at(i + 1, j)]) {
                ops.push(('+', bm[j]));
                j += 1;
            } else {
                ops.push(('-', am[i]));
                i += 1;
            }
        }
    } else {
        ops.extend(am.iter().map(|l| ('-', *l)));
        ops.extend(bm.iter().map(|l| ('+', *l)));
    }
    // Each hunk is headed by the item it falls in: the card or endpoint list (its `///` line or
    // `model:`) and, inside an endpoint list, the endpoint's `tag:`.
    let mut item = String::new();
    let mut tag = String::new();
    let note = |l: &str, item: &mut String, tag: &mut String| {
        let t = l.trim();
        if t.starts_with("/// ") || t.starts_with("model: ") {
            *item = t.to_owned();
            tag.clear();
        } else if t.starts_with("tag: ") {
            *tag = t.to_owned();
        }
    };
    for l in &a[..pre] {
        note(l, &mut item, &mut tag);
    }
    let mut out = String::new();
    let mut last_header = String::new();
    for (k, l) in ops {
        if k != '+' {
            note(l, &mut item, &mut tag);
        }
        if k == ' ' {
            continue;
        }
        let header = if tag.is_empty() {
            item.clone()
        } else {
            format!("{item} {tag}")
        };
        if header != last_header {
            out.push_str(&format!("@@ {header}\n"));
            last_header = header;
        }
        out.push(k);
        out.push_str(l);
        out.push('\n');
    }
    Some(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    #[test]
    fn hunks_name_their_endpoint() {
        let old = "/// `a/b` (x).\n    OrEndpoint {\n        tag: \"t1\",\n        card: 1,\n";
        let new = "/// `a/b` (x).\n    OrEndpoint {\n        tag: \"t1\",\n        card: 2,\n";
        let d = super::lines(old, new).unwrap();
        assert_eq!(
            d,
            "@@ /// `a/b` (x). tag: \"t1\",\n-        card: 1,\n+        card: 2,\n"
        );
        assert!(super::lines(old, old).is_none());
    }
}
