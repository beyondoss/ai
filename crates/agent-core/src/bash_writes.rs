//! The files a `bash` command writes, recovered from the command text alone — so compaction's
//! modified-files provenance covers edits made through the shell, not only through `write`/`edit`.
//!
//! In the long-context eval, Claude Sonnet 5.5 made 2 `write`/`edit` calls across 17 runs and did
//! every edit through `bash` (`python - <<'EOF' … open(p, 'w').write(s)`, `sed -i`, redirects), so
//! its `<modified-files>` was always empty and 19 of its 23 summaries missed files it had changed.
//!
//! Deliberately a parser, not a filesystem diff: it stays pure (the provenance is derived from the
//! transcript, so it is reproducible on resume and costs nothing per call), at the price of missing
//! writes it cannot see in the text (a script file run by name, a path computed at runtime). It
//! recognizes:
//! - shell redirects (`> f`, `>> f`, `>f`; not `2>`, `&>`, `>&2` or `/dev/…`), `tee [-a] f…`,
//!   `sed -i … f…`, `perl -i … f…`, and the destination of `mv`/`cp`;
//! - Python in a heredoc or `python -c`: `open(X, 'w'|'a'|…)`, `Path(X).write_text/write_bytes(…)`,
//!   `X.write_text(…)`, where `X` is a string literal or a name bound to one (`p = 'a.py'`,
//!   `p = Path('a.py')`, `for f in ['a.py', 'b.py']:`).
//!
//! A relative path is joined onto the directory of the last `cd` before it, lexically. Paths under
//! `/dev`, `/proc` and `/tmp` (scratch, not workspace files) are left out.

/// The files `command` writes, in first-seen order, deduped. See the module docs for what counts.
pub fn written_paths(command: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cwd: Option<String> = None;
    let (shell, scripts) = split_heredocs(command);
    let mut push = |cwd: &Option<String>, path: &str| {
        let path = path.trim();
        if path.is_empty() || path.starts_with('$') || path.contains('*') {
            return;
        }
        let joined = match cwd {
            Some(dir) if !path.starts_with('/') && !path.starts_with('~') => {
                normalize(&format!("{dir}/{path}"))
            }
            _ => normalize(path),
        };
        if ["/dev/", "/proc/", "/tmp/"]
            .iter()
            .any(|p| joined.starts_with(p))
            || joined == "/dev/null"
        {
            return;
        }
        if !out.contains(&joined) {
            out.push(joined);
        }
    };
    // Heredoc bodies attach to the shell segment that opened them; scan each in the cwd in effect
    // at that point.
    let mut script_at: Vec<(usize, &str)> = scripts;
    script_at.reverse();
    for (line_no, line) in shell.lines().enumerate() {
        for segment in split_segments(line) {
            let words = shell_words(segment);
            let Some(first) = words.first() else {
                continue;
            };
            redirect_targets(&words, |p| push(&cwd, p));
            match first.as_str() {
                "cd" => {
                    if let Some(dir) = words.get(1) {
                        cwd = Some(match &cwd {
                            Some(c) if !dir.starts_with('/') => normalize(&format!("{c}/{dir}")),
                            _ => normalize(dir),
                        });
                    }
                }
                "tee" => words[1..]
                    .iter()
                    .filter(|w| !w.starts_with('-') && !is_redirect(w))
                    .for_each(|w| push(&cwd, w)),
                "sed" | "perl" if words.iter().any(|w| is_in_place(first, w)) => {
                    in_place_files(&words).for_each(|w| push(&cwd, w))
                }
                "mv" | "cp" => {
                    if let Some(dst) = words[1..]
                        .iter()
                        .filter(|w| !w.starts_with('-') && !is_redirect(w))
                        .nth(1)
                    {
                        push(&cwd, dst);
                    }
                }
                cmd if cmd.starts_with("python") => {
                    if let Some(i) = words.iter().position(|w| w == "-c")
                        && let Some(code) = words.get(i + 1)
                    {
                        python_writes(code, |p| push(&cwd, p));
                    }
                }
                _ => {}
            }
        }
        while let Some(&(at, body)) = script_at.last() {
            if at != line_no {
                break;
            }
            python_writes(body, |p| push(&cwd, p));
            script_at.pop();
        }
    }
    out
}

/// The command with every heredoc body removed, plus each body and the (0-based) line of the
/// remaining text that opened it.
fn split_heredocs(command: &str) -> (String, Vec<(usize, &str)>) {
    let mut shell = String::new();
    let mut bodies = Vec::new();
    let mut lines = command.split_inclusive('\n');
    let mut offset = 0;
    let mut shell_lines = 0;
    while let Some(line) = lines.next() {
        offset += line.len();
        shell.push_str(line);
        let delimiter = heredoc_delimiter(line);
        let opened_at = shell_lines;
        shell_lines += 1;
        if let Some(delim) = delimiter {
            let start = offset;
            let mut end = offset;
            for body_line in lines.by_ref() {
                offset += body_line.len();
                if body_line.trim() == delim {
                    break;
                }
                end = offset;
            }
            bodies.push((opened_at, &command[start..end]));
        }
    }
    if !shell.ends_with('\n') {
        shell.push('\n');
    }
    (shell, bodies)
}

/// The delimiter word of a `<<` / `<<-` heredoc opened on `line` (quotes stripped), if any. A `<<<`
/// here-string opens nothing.
fn heredoc_delimiter(line: &str) -> Option<String> {
    let at = line.find("<<")?;
    let rest = &line[at + 2..];
    if rest.starts_with('<') {
        return None;
    }
    let rest = rest.strip_prefix('-').unwrap_or(rest).trim_start();
    let word: String = rest
        .chars()
        .take_while(|c| !c.is_whitespace() && !matches!(c, ';' | '&' | '|' | ')' | '>'))
        .collect();
    let word = word.trim_matches(|c| c == '\'' || c == '"').to_string();
    (!word.is_empty()).then_some(word)
}

/// A shell line split at `&&`, `||`, `;` and `|`, outside quotes.
fn split_segments(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut start, mut quote) = (0, None::<char>);
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '\'' || c == '"' => quote = Some(c),
            None if c == ';' || c == '|' || (c == '&' && bytes.get(i + 1) == Some(&b'&')) => {
                out.push(&line[start..i]);
                i += if bytes.get(i + 1) == Some(&bytes[i]) {
                    2
                } else {
                    1
                };
                start = i;
                continue;
            }
            None => {}
        }
        i += 1;
    }
    out.push(&line[start..]);
    out
}

/// Whitespace-separated words, quotes removed (no expansion).
fn shell_words(segment: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let (mut quote, mut any) = (None::<char>, false);
    for c in segment.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => cur.push(c),
            None if c == '\'' || c == '"' => {
                quote = Some(c);
                any = true;
            }
            None if c.is_whitespace() => {
                if any || !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
                any = false;
            }
            None => cur.push(c),
        }
    }
    if any || !cur.is_empty() {
        words.push(cur);
    }
    words
}

fn is_redirect(word: &str) -> bool {
    word.starts_with('>') || word.starts_with("2>") || word.starts_with("&>")
}

/// Every `>`/`>>` target in `words`: `> f`, `>f`, `>> f`. Not a descriptor redirect (`2>`, `&>`,
/// `>&2`).
fn redirect_targets(words: &[String], mut push: impl FnMut(&str)) {
    for (i, w) in words.iter().enumerate() {
        let rest = w
            .strip_prefix(">>")
            .or_else(|| w.strip_prefix('>'))
            .or_else(|| w.strip_prefix("1>>"))
            .or_else(|| w.strip_prefix("1>"));
        let Some(rest) = rest else {
            continue;
        };
        if rest.starts_with('&') {
            continue;
        }
        if rest.is_empty() {
            if let Some(next) = words.get(i + 1) {
                push(next);
            }
        } else {
            push(rest);
        }
    }
}

fn is_in_place(cmd: &str, word: &str) -> bool {
    match cmd {
        "sed" => {
            word == "--in-place" || (word.starts_with("-i") && !word.starts_with("-i=")) || {
                word.starts_with('-') && !word.starts_with("--") && word[1..].contains('i')
            }
        }
        _ => word.starts_with('-') && !word.starts_with("--") && word[1..].contains('i'),
    }
}

/// The file operands of an in-place `sed`/`perl`: the positional words, minus the script — the
/// first positional, unless an `-e`/`-f` flag gave it.
fn in_place_files(words: &[String]) -> impl Iterator<Item = &String> {
    let mut positional = Vec::new();
    let mut script_given = false;
    let mut skip = false;
    for w in &words[1..] {
        if skip {
            skip = false;
            continue;
        }
        if is_redirect(w) {
            break;
        }
        if w == "-e" || w == "-f" || w == "--expression" {
            script_given = true;
            skip = true;
        } else if w.starts_with('-') && w.len() > 1 {
            // `-pe`, `-ne`, `-0777pe`: a bundled `e` takes the script from the next word.
            if !w.starts_with("--") && w.ends_with('e') {
                script_given = true;
                skip = true;
            }
        } else {
            positional.push(w);
        }
    }
    let skip_first = usize::from(!script_given);
    positional.into_iter().skip(skip_first)
}

/// A lexical token of (roughly) Python: enough to match the write shapes in the module docs.
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Punct(char),
}

fn python_tokens(code: &str) -> Vec<Tok> {
    let chars: Vec<char> = code.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '\'' || c == '"' {
            // A triple-quoted string is code or prose, not a path: skip it whole.
            if chars.get(i + 1) == Some(&c) && chars.get(i + 2) == Some(&c) {
                i += 3;
                while i < chars.len()
                    && !(chars[i] == c
                        && chars.get(i + 1) == Some(&c)
                        && chars.get(i + 2) == Some(&c))
                {
                    i += 1;
                }
                i += 3;
                continue;
            }
            let mut s = String::new();
            i += 1;
            while i < chars.len() && chars[i] != c && chars[i] != '\n' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 1;
                }
                s.push(chars[i]);
                i += 1;
            }
            i += 1;
            toks.push(Tok::Str(s));
            continue;
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            // A string prefix (`r'…'`, `f"…"`, `b'…'`): the next token is the literal.
            if word.len() <= 2
                && word.chars().all(|c| "rRbBfFuU".contains(c))
                && matches!(chars.get(i), Some('\'' | '"'))
            {
                continue;
            }
            toks.push(Tok::Ident(word));
            continue;
        } else if !c.is_whitespace() {
            toks.push(Tok::Punct(c));
        }
        i += 1;
    }
    toks
}

/// The paths a Python snippet writes (module docs), reported through `push`.
fn python_writes(code: &str, mut push: impl FnMut(&str)) {
    let t = python_tokens(code);
    let mut bound: Vec<(String, Vec<String>)> = Vec::new();
    let lookup = |bound: &[(String, Vec<String>)], name: &str| -> Vec<String> {
        bound
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    };
    let ident = |i: usize| match t.get(i) {
        Some(Tok::Ident(s)) => Some(s.as_str()),
        _ => None,
    };
    let string = |i: usize| match t.get(i) {
        Some(Tok::Str(s)) => Some(s.as_str()),
        _ => None,
    };
    let punct = |i: usize, c: char| t.get(i) == Some(&Tok::Punct(c));
    // The value at `i`: a literal, a bound name, or `Path(literal-or-name)`.
    let value = |bound: &[(String, Vec<String>)], i: usize| -> Vec<String> {
        if let Some(s) = string(i) {
            return vec![s.to_string()];
        }
        match ident(i) {
            Some("Path" | "pathlib") if punct(i + 1, '(') => match (string(i + 2), ident(i + 2)) {
                (Some(s), _) => vec![s.to_string()],
                (_, Some(n)) => lookup(bound, n),
                _ => Vec::new(),
            },
            Some(n) => lookup(bound, n),
            None => Vec::new(),
        }
    };
    for i in 0..t.len() {
        // `name = <value>`, not `==`, and only a whole value: `p = D + path` or `p = s.strip()` binds
        // nothing, since the value is not the path.
        if let Some(name) = ident(i)
            && punct(i + 1, '=')
            && !punct(i + 2, '=')
            && (i == 0 || !punct(i - 1, '.'))
        {
            let len = if matches!(ident(i + 2), Some("Path" | "pathlib")) && punct(i + 3, '(') {
                4
            } else {
                1
            };
            let operand_follows = ['+', '%', '.', '[', '*', '(']
                .iter()
                .any(|&c| punct(i + 2 + len, c));
            let v = value(&bound, i + 2);
            if !v.is_empty() && !operand_follows {
                bound.push((name.to_string(), v));
            }
        }
        // `for name in [lit, lit]` / `(lit, lit)`.
        if ident(i) == Some("for")
            && let Some(name) = ident(i + 1)
            && ident(i + 2) == Some("in")
            && (punct(i + 3, '[') || punct(i + 3, '('))
        {
            let mut items = Vec::new();
            let mut j = i + 4;
            while let Some(s) = string(j) {
                items.push(s.to_string());
                if !punct(j + 1, ',') {
                    break;
                }
                j += 2;
            }
            if !items.is_empty() {
                bound.push((name.to_string(), items));
            }
        }
        // `open(X, 'w')`, `open(X, mode='a')`.
        if ident(i) == Some("open") && punct(i + 1, '(') && punct(i + 3, ',') {
            let mode = string(i + 4).or_else(|| {
                (ident(i + 4) == Some("mode") && punct(i + 5, '='))
                    .then(|| string(i + 6))
                    .flatten()
            });
            if mode.is_some_and(|m| m.contains(['w', 'a', 'x', '+'])) {
                value(&bound, i + 2).iter().for_each(|p| push(p));
            }
        }
        // `….write_text(` / `….write_bytes(` on `Path(X)` or on a name.
        if punct(i, '.') && matches!(ident(i + 1), Some("write_text" | "write_bytes")) && i > 0 {
            if punct(i - 1, ')') {
                // `Path(X).write_text`: X sits two tokens back, after `Path(`.
                if i >= 3 && punct(i - 3, '(') && matches!(ident(i - 4), Some("Path")) {
                    value(&bound, i - 2).iter().for_each(|p| push(p));
                }
            } else if let Some(n) = ident(i - 1) {
                lookup(&bound, n).iter().for_each(|p| push(p));
            }
        }
    }
}

/// `a/./b/../c` → `a/c`, lexically (no filesystem access); keeps a leading `/` and leading `..`.
fn normalize(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." if parts.last().is_some_and(|p| *p != "..") => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    let joined = parts.join("/");
    if absolute {
        format!("/{joined}")
    } else {
        joined
    }
}

#[cfg(test)]
mod tests {
    use super::written_paths;

    #[test]
    fn a_python_heredoc_that_rewrites_a_named_file() {
        // The shape behind most of Sonnet 5.5's edits in the eval.
        let cmd = "cd click && python - <<'EOF'\np='src/click/_termui_impl.py'\ns=open(p).read()\ns=s.replace('a','b')\nopen(p,'w').write(s)\nEOF\npython -m pytest tests -q 2>&1 | tail -3";
        assert_eq!(written_paths(cmd), ["click/src/click/_termui_impl.py"]);
    }

    #[test]
    fn a_heredoc_loop_over_literal_files_and_a_read_only_open() {
        let cmd = "cd toolz/toolz; python - <<'EOF'\nimport re\nsrc = open('README.md').read()\nfor f in ['_signatures.py','functoolz.py']:\n    L=open(f).read().split('\\n')\n    open(f,'w').write('\\n'.join(L))\nEOF\npython -c \"import toolz\"";
        assert_eq!(
            written_paths(cmd),
            ["toolz/toolz/_signatures.py", "toolz/toolz/functoolz.py"]
        );
    }

    #[test]
    fn pathlib_writes_through_a_literal_or_a_bound_name() {
        let cmd = "python3 - <<EOF\nfrom pathlib import Path\nPath('a/b.py').write_text('x')\nq = Path(\"c.py\")\nq.write_text(q.read_text().replace('1', '2'))\nEOF";
        assert_eq!(written_paths(cmd), ["a/b.py", "c.py"]);
    }

    #[test]
    fn a_python_dash_c_one_liner() {
        let cmd = "python -c \"open('setup.cfg', 'a').write('[x]\\n')\"";
        assert_eq!(written_paths(cmd), ["setup.cfg"]);
    }

    #[test]
    fn shell_writes_redirects_tee_sed_perl_mv_cp() {
        assert_eq!(written_paths("echo hi > notes.txt"), ["notes.txt"]);
        assert_eq!(written_paths("printf x >>log.md; cat a"), ["log.md"]);
        assert_eq!(
            written_paths("cat a | tee -a out1.txt out2.txt"),
            ["out1.txt", "out2.txt"]
        );
        assert_eq!(
            written_paths("cd pkg && sed -i 's/a/b/' src/m.py src/n.py"),
            ["pkg/src/m.py", "pkg/src/n.py"]
        );
        assert_eq!(
            written_paths("sed -i.bak -e 's/a/b/' -e 's/c/d/' x.py"),
            ["x.py"]
        );
        assert_eq!(
            written_paths("perl -pi -e 's/foo/bar/g' lib/a.pm"),
            ["lib/a.pm"]
        );
        assert_eq!(written_paths("mv -f old.py new.py"), ["new.py"]);
        assert_eq!(written_paths("cp a.py b/"), ["b"]);
        assert_eq!(
            written_paths("cat > src/conf.toml <<'EOF'\n[a]\nb = 1\nEOF"),
            ["src/conf.toml"]
        );
    }

    #[test]
    fn reads_scratch_and_descriptor_redirects_are_not_writes() {
        assert!(written_paths("python -m pytest -q 2>&1 | tail -5").is_empty());
        assert!(written_paths("pytest > /dev/null 2>&1").is_empty());
        assert!(written_paths("pytest -q > /tmp/out.log; cat /tmp/out.log").is_empty());
        assert!(written_paths("echo err >&2").is_empty());
        assert!(written_paths("sed -n '1,20p' src/a.py").is_empty());
        assert!(written_paths("grep -rn 'x > y' src").is_empty());
        assert!(written_paths("python - <<'EOF'\nprint(open('a.py').read())\nEOF").is_empty());
        assert!(written_paths("cat <<< 'x > y.txt'").is_empty());
        // A computed path is not the literal it starts from (seen in the eval: `p = D + path`).
        assert!(
            written_paths("python - <<'EOF'\nD = 'src/pkg/'\ndef rep(path):\n    p = D + path\n    open(p, 'w').write('x')\nEOF")
                .is_empty()
        );
    }

    /// Not a test: runs the parser over a corpus of real commands for offline measurement.
    /// `BASH_WRITES_CORPUS=in.jsonl BASH_WRITES_OUT=out.jsonl cargo test -p beyond-agent-core
    /// --lib -- --ignored bash_writes_over_a_corpus`; each input line is `{"id": …, "command": "…"}`,
    /// each output line adds `"paths": […]`.
    #[test]
    #[ignore]
    fn bash_writes_over_a_corpus() {
        let (Ok(input), Ok(output)) = (
            std::env::var("BASH_WRITES_CORPUS"),
            std::env::var("BASH_WRITES_OUT"),
        ) else {
            return;
        };
        let mut out = String::new();
        for line in std::fs::read_to_string(input).unwrap().lines() {
            let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
            let paths = written_paths(v["command"].as_str().unwrap_or_default());
            v["paths"] = serde_json::json!(paths);
            out.push_str(&v.to_string());
            out.push('\n');
        }
        std::fs::write(output, out).unwrap();
    }

    #[test]
    fn cd_paths_join_and_normalize_lexically() {
        assert_eq!(
            written_paths("cd a/b && cd ../c && echo x > ./d.txt"),
            ["a/c/d.txt"]
        );
        assert_eq!(written_paths("cd /w && echo x > e.txt"), ["/w/e.txt"]);
        assert_eq!(
            written_paths("cd /w && echo x > /abs/f.txt"),
            ["/abs/f.txt"]
        );
    }
}
