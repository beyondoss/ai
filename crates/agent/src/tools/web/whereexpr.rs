//! The `--where` filter expression evaluator for `extract` mode — ax's `price > 100 && name ~ /^foo/i`.
//!
//! A small hand-rolled recursive-descent parser + evaluator over one extracted row (a map of
//! field→string). Deliberately not a general expression crate: the grammar is tiny and closed, and
//! hand-rolling keeps it dependency-light and panic-free — a malformed expression is a `ToolError`, and
//! an unknown field is simply an empty value, never a crash.
//!
//! Grammar (loosest to tightest binding):
//! ```text
//!   or     := and ("||" and)*
//!   and    := cmp ("&&" cmp)*
//!   cmp    := term (op term)?        op ∈ { == != < <= > >= ~ }
//!   term   := ident | number | string | regex | "(" or ")"
//!   regex  := "/" pattern "/" flags  (flag `i` = case-insensitive)
//! ```
//! A bare `ident` used as a whole `cmp` (no operator) is truthy when its field value is non-empty —
//! `where 'href'` keeps rows that have an `href`.
//!
//! A row is not passed as a `HashMap`: [`Filter::matches`] takes a `get(field) -> Option<&str>`
//! lookup so the caller can resolve a field name against a fixed positional row (name→index computed
//! once) without cloning field names into a per-row map. `None` means "no such column" (a bare-field
//! truthiness test fails); `Some("")` means the column exists but is empty.

use regex::Regex;

/// A parsed filter, reusable across rows (the regex is compiled once).
pub struct Filter {
    root: Expr,
}

impl Filter {
    /// Parse `src`. `Err` names the problem for the model to correct.
    pub fn parse(src: &str) -> Result<Self, String> {
        let tokens = lex(src)?;
        let mut p = Parser { tokens, pos: 0 };
        let root = p.parse_or()?;
        if p.pos != p.tokens.len() {
            return Err(format!(
                "unexpected trailing input in `where` at token {}",
                p.pos
            ));
        }
        Ok(Self { root })
    }

    /// Whether a row passes the filter. `get(field)` resolves a field name to its value: `None` if the
    /// row has no such column, `Some(value)` otherwise. The caller keeps the row positional and hands us
    /// an index-based lookup — no per-row map, no field-name clones.
    pub fn matches<'r>(&self, get: &impl Fn(&str) -> Option<&'r str>) -> bool {
        self.root.eval(get)
    }
}

enum Expr {
    Or(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Cmp {
        field: String,
        op: Op,
        rhs: Operand,
    },
    /// A bare field reference: truthy when non-empty.
    Truthy(String),
}

#[derive(Clone, Copy)]
enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Match,
}

enum Operand {
    Str(String),
    Num(f64),
    Regex(Regex),
}

impl Expr {
    fn eval<'r>(&self, get: &impl Fn(&str) -> Option<&'r str>) -> bool {
        match self {
            Expr::Or(a, b) => a.eval(get) || b.eval(get),
            Expr::And(a, b) => a.eval(get) && b.eval(get),
            Expr::Truthy(field) => get(field).is_some_and(|v| !v.is_empty()),
            Expr::Cmp { field, op, rhs } => {
                let lhs = get(field).unwrap_or("");
                match (op, rhs) {
                    (Op::Match, Operand::Regex(re)) => re.is_match(lhs),
                    // A `~` against a non-regex right-hand side never matches (a parse-time guard also
                    // prevents this, but be defensive).
                    (Op::Match, _) => false,
                    (op, Operand::Num(n)) => match lhs.trim().parse::<f64>() {
                        Ok(l) => cmp_num(*op, l, *n),
                        Err(_) => false, // non-numeric field vs a number never orders
                    },
                    (op, Operand::Str(s)) => {
                        // Numeric compare when both sides look numeric, else lexical.
                        match (lhs.trim().parse::<f64>(), s.trim().parse::<f64>()) {
                            (Ok(l), Ok(r)) => cmp_num(*op, l, r),
                            _ => cmp_str(*op, lhs, s),
                        }
                    }
                    (_, Operand::Regex(_)) => false, // an ordering op against a regex is meaningless
                }
            }
        }
    }
}

fn cmp_num(op: Op, l: f64, r: f64) -> bool {
    match op {
        Op::Eq => l == r,
        Op::Ne => l != r,
        Op::Lt => l < r,
        Op::Le => l <= r,
        Op::Gt => l > r,
        Op::Ge => l >= r,
        Op::Match => false,
    }
}

fn cmp_str(op: Op, l: &str, r: &str) -> bool {
    match op {
        Op::Eq => l == r,
        Op::Ne => l != r,
        Op::Lt => l < r,
        Op::Le => l <= r,
        Op::Gt => l > r,
        Op::Ge => l >= r,
        Op::Match => false,
    }
}

// ---- lexer ---------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Str(String),
    Regex(String, String), // (pattern, flags)
    Op(&'static str),      // == != < <= > >= ~ && ||
    LParen,
    RParen,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '&' | '|' => {
                if i + 1 < chars.len() && chars[i + 1] == c {
                    out.push(Tok::Op(if c == '&' { "&&" } else { "||" }));
                    i += 2;
                } else {
                    return Err(format!("expected `{c}{c}` in `where`"));
                }
            }
            '=' | '!' | '<' | '>' => {
                let two = i + 1 < chars.len() && chars[i + 1] == '=';
                let op = match (c, two) {
                    ('=', true) => "==",
                    ('!', true) => "!=",
                    ('<', true) => "<=",
                    ('>', true) => ">=",
                    ('<', false) => "<",
                    ('>', false) => ">",
                    ('=', false) => return Err("use `==` for equality in `where`".into()),
                    ('!', false) => return Err("`!` must be `!=` in `where`".into()),
                    _ => unreachable!(),
                };
                out.push(Tok::Op(op));
                i += if two { 2 } else { 1 };
            }
            '~' => {
                out.push(Tok::Op("~"));
                i += 1;
            }
            '"' | '\'' => {
                let quote = c;
                i += 1;
                let mut s = String::new();
                while i < chars.len() && chars[i] != quote {
                    s.push(chars[i]);
                    i += 1;
                }
                if i >= chars.len() {
                    return Err("unterminated string in `where`".into());
                }
                i += 1; // closing quote
                out.push(Tok::Str(s));
            }
            '/' => {
                // A regex literal: /pattern/flags. `\/` escapes a slash inside the pattern.
                i += 1;
                let mut pat = String::new();
                while i < chars.len() && chars[i] != '/' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        pat.push(chars[i]);
                        pat.push(chars[i + 1]);
                        i += 2;
                        continue;
                    }
                    pat.push(chars[i]);
                    i += 1;
                }
                if i >= chars.len() {
                    return Err("unterminated regex in `where` (missing closing `/`)".into());
                }
                i += 1; // closing slash
                let mut flags = String::new();
                while i < chars.len() && chars[i].is_ascii_alphabetic() {
                    flags.push(chars[i]);
                    i += 1;
                }
                out.push(Tok::Regex(pat, flags));
            }
            c if c.is_ascii_digit()
                || (c == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) =>
            {
                let mut n = String::new();
                if c == '-' {
                    n.push('-');
                    i += 1;
                }
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    n.push(chars[i]);
                    i += 1;
                }
                let val = n
                    .parse::<f64>()
                    .map_err(|_| format!("bad number `{n}` in `where`"))?;
                out.push(Tok::Num(val));
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut id = String::new();
                while i < chars.len()
                    && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '-')
                {
                    id.push(chars[i]);
                    i += 1;
                }
                out.push(Tok::Ident(id));
            }
            other => return Err(format!("unexpected character `{other}` in `where`")),
        }
    }
    Ok(out)
}

// ---- parser --------------------------------------------------------------------------------------

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Tok::Op("||"))) {
            self.pos += 1;
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_cmp()?;
        while matches!(self.peek(), Some(Tok::Op("&&"))) {
            self.pos += 1;
            let right = self.parse_cmp()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Expr, String> {
        // Parenthesized sub-expression.
        if matches!(self.peek(), Some(Tok::LParen)) {
            self.pos += 1;
            let inner = self.parse_or()?;
            match self.peek() {
                Some(Tok::RParen) => {
                    self.pos += 1;
                    return Ok(inner);
                }
                _ => return Err("missing `)` in `where`".into()),
            }
        }

        // The left side of a comparison must be a field reference — that is what a row is keyed by.
        let field = match self.peek() {
            Some(Tok::Ident(name)) => {
                let name = name.clone();
                self.pos += 1;
                name
            }
            _ => return Err("expected a field name in `where`".into()),
        };

        let op = match self.peek() {
            Some(Tok::Op(o)) if *o != "&&" && *o != "||" => {
                let op = op_from(o)?;
                self.pos += 1;
                op
            }
            // No operator: a bare field is a truthiness test.
            _ => return Ok(Expr::Truthy(field)),
        };

        let rhs = match self.tokens.get(self.pos) {
            Some(Tok::Str(s)) => Operand::Str(s.clone()),
            Some(Tok::Num(n)) => Operand::Num(*n),
            Some(Tok::Regex(pat, flags)) => {
                let re = build_regex(pat, flags)?;
                Operand::Regex(re)
            }
            Some(Tok::Ident(id)) => Operand::Str(id.clone()), // bare word compared as a string
            _ => return Err("expected a value on the right of the operator in `where`".into()),
        };
        self.pos += 1;

        if matches!(op, Op::Match) && !matches!(rhs, Operand::Regex(_)) {
            return Err("`~` needs a /regex/ on its right in `where`".into());
        }
        Ok(Expr::Cmp { field, op, rhs })
    }
}

fn op_from(o: &str) -> Result<Op, String> {
    Ok(match o {
        "==" => Op::Eq,
        "!=" => Op::Ne,
        "<" => Op::Lt,
        "<=" => Op::Le,
        ">" => Op::Gt,
        ">=" => Op::Ge,
        "~" => Op::Match,
        other => return Err(format!("unknown operator `{other}` in `where`")),
    })
}

fn build_regex(pat: &str, flags: &str) -> Result<Regex, String> {
    let mut builder = regex::RegexBuilder::new(pat);
    for f in flags.chars() {
        match f {
            'i' => {
                builder.case_insensitive(true);
            }
            's' => {
                builder.dot_matches_new_line(true);
            }
            'm' => {
                builder.multi_line(true);
            }
            other => return Err(format!("unknown regex flag `{other}` in `where`")),
        }
    }
    builder
        .build()
        .map_err(|e| format!("invalid regex `/{pat}/`: {e}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn row(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn passes(expr: &str, pairs: &[(&str, &str)]) -> bool {
        let row = row(pairs);
        Filter::parse(expr)
            .unwrap()
            .matches(&|field| row.get(field).map(String::as_str))
    }

    #[test]
    fn numeric_comparisons() {
        assert!(passes("price > 100", &[("price", "150")]));
        assert!(!passes("price > 100", &[("price", "50")]));
        assert!(passes("price <= 100", &[("price", "100")]));
        assert!(passes("price == 100", &[("price", "100")]));
        assert!(passes("qty != 0", &[("qty", "3")]));
        // A non-numeric field never orders against a number.
        assert!(!passes("price > 100", &[("price", "cheap")]));
    }

    #[test]
    fn string_comparisons() {
        assert!(passes("name == \"foo\"", &[("name", "foo")]));
        assert!(passes("name == 'foo'", &[("name", "foo")]));
        assert!(!passes("name == \"foo\"", &[("name", "bar")]));
        assert!(passes("name != \"foo\"", &[("name", "bar")]));
    }

    #[test]
    fn regex_match_with_flags() {
        assert!(passes("name ~ /^foo/", &[("name", "foobar")]));
        assert!(!passes("name ~ /^foo/", &[("name", "barfoo")]));
        assert!(passes("name ~ /^FOO/i", &[("name", "foobar")]));
        assert!(!passes("name ~ /^FOO/", &[("name", "foobar")]));
    }

    #[test]
    fn logical_operators_and_precedence() {
        // && binds tighter than ||.
        assert!(passes(
            "a == 1 && b == 2 || c == 3",
            &[("a", "9"), ("b", "9"), ("c", "3")]
        ));
        assert!(passes(
            "a == 1 && b == 2 || c == 3",
            &[("a", "1"), ("b", "2"), ("c", "9")]
        ));
        assert!(!passes("a == 1 && b == 2", &[("a", "1"), ("b", "9")]));
    }

    #[test]
    fn parentheses_override_precedence() {
        assert!(!passes(
            "a == 1 && (b == 2 || c == 3)",
            &[("a", "1"), ("b", "9"), ("c", "9")]
        ));
        assert!(passes(
            "a == 1 && (b == 2 || c == 3)",
            &[("a", "1"), ("b", "9"), ("c", "3")]
        ));
    }

    #[test]
    fn a_bare_field_is_a_truthiness_test() {
        assert!(passes("href", &[("href", "https://x")]));
        assert!(!passes("href", &[("href", "")]));
        assert!(!passes("href", &[("title", "x")])); // absent field
    }

    #[test]
    fn malformed_expressions_are_errors_not_panics() {
        for bad in [
            "price >",
            "== 5",
            "name ~ \"foo\"", // ~ needs a regex
            "name ~ /unterminated",
            "a == 1 &",
            "(a == 1",
            "name = 5", // single =
            "@#$",
        ] {
            assert!(Filter::parse(bad).is_err(), "`{bad}` should not parse");
        }
    }
}
