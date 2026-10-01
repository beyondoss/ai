//! `verify`: the claim registry's status and gate.
//!
//! Status is never written by hand. It is computed from three sources:
//!
//! - `verify/claims.toml`: every claim the gateway makes.
//! - `verify/defects.toml`: suspected and confirmed defects, each with a lifecycle state.
//! - The tests themselves: a test claims a claim with a `/// claim: ID[, ID…]` doc line, and a
//!   defect with `/// defect: ID`. Results come from nextest's JUnit report.
//!
//! ```text
//! verify filter                       nextest filter expression for every tagged test
//! verify status [--junit P] [--json]  per-claim status and the summary
//! verify gate   [--junit P]           fail on any registry/tag/result inconsistency
//! ```
//!
//! A test tagged with a `reproduced` defect asserts the *correct* behavior, is `#[ignore]`d so the
//! normal suite stays green, and must fail when run. When it starts passing, the gate stops until
//! the defect is marked `fixed` (or `refuted`), so the ledger cannot drift from the code.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Deserialize;

#[derive(Deserialize)]
struct Claims {
    claim: Vec<Claim>,
}

#[derive(Deserialize)]
struct Claim {
    id: String,
    priority: String,
    group: String,
    text: String,
    layer: String,
    #[serde(default)]
    clients: Vec<String>,
    touches: Vec<String>,
    #[serde(default)]
    defects: Vec<String>,
}

#[derive(Deserialize)]
struct Defects {
    defect: Vec<Defect>,
}

#[derive(Deserialize)]
struct Defect {
    id: String,
    severity: String,
    title: String,
    claims: Vec<String>,
    state: String,
    test: String,
    fix: String,
    note: String,
}

/// One test function carrying `claim:` / `defect:` tags.
struct Tagged {
    name: String,
    /// nextest binary id the test lives in, e.g. `beyond-ai::e2e` or `beyond-ai` (lib).
    binary: String,
    file: PathBuf,
    line: usize,
    ignored: bool,
    claims: Vec<String>,
    defects: Vec<String>,
}

/// `(nextest binary id, test name) → outcome`.
type Results = BTreeMap<(String, String), Outcome>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Passed,
    Failed,
    Skipped,
}

const STATES: [&str; 5] = ["suspected", "reproduced", "fixed", "refuted", "accepted"];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = repo_root();
    let result = match args.first().map(String::as_str) {
        Some("filter") => load(&root).map(|ctx| {
            println!("{}", filter_expr(&ctx.tagged));
            true
        }),
        Some("status") => load(&root).map(|ctx| status(&ctx, &args, &root)),
        Some("gate") => load(&root).map(|ctx| gate(&ctx, &args, &root)),
        _ => Err("usage: verify <filter|status|gate> [--junit PATH] [--json]".to_owned()),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("verify: {e}");
            ExitCode::from(2)
        }
    }
}

fn repo_root() -> PathBuf {
    // crates/verify → repo root. Running from anywhere in the repo works.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Ctx {
    claims: Vec<Claim>,
    defects: Vec<Defect>,
    tagged: Vec<Tagged>,
}

fn load(root: &Path) -> Result<Ctx, String> {
    let read =
        |p: &str| std::fs::read_to_string(root.join(p)).map_err(|e| format!("reading {p}: {e}"));
    let claims: Claims =
        toml::from_str(&read("verify/claims.toml")?).map_err(|e| format!("claims.toml: {e}"))?;
    let defects: Defects =
        toml::from_str(&read("verify/defects.toml")?).map_err(|e| format!("defects.toml: {e}"))?;
    let tagged = scan_tags(root)?;
    Ok(Ctx {
        claims: claims.claim,
        defects: defects.defect,
        tagged,
    })
}

/// Every `.rs` file under `crates/*/src` and `crates/*/tests`, scanned for tag lines.
fn scan_tags(root: &Path) -> Result<Vec<Tagged>, String> {
    let mut out = Vec::new();
    let crates = std::fs::read_dir(root.join("crates")).map_err(|e| e.to_string())?;
    for krate in crates.flatten() {
        let dir = krate.path();
        let Some(pkg) = package_name(&dir) else {
            continue;
        };
        for sub in ["src", "tests"] {
            let mut files = Vec::new();
            walk(&dir.join(sub), &mut files);
            for file in files {
                let binary = if sub == "tests" {
                    // tests/foo.rs and tests/foo/main.rs are binary `pkg::foo`; tests/common/* is
                    // shared code compiled into every binary, never its own.
                    let rel = file.strip_prefix(dir.join("tests")).unwrap_or(&file);
                    let mut parts = rel.components();
                    let first = parts
                        .next()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .unwrap_or_default();
                    format!("{pkg}::{}", first.trim_end_matches(".rs"))
                } else {
                    pkg.clone()
                };
                scan_file(&file, &binary, &mut out);
            }
        }
    }
    out.sort_by(|a, b| (&a.binary, &a.name).cmp(&(&b.binary, &b.name)));
    Ok(out)
}

fn package_name(dir: &Path) -> Option<String> {
    let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).ok()?;
    let mut in_package = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
        } else if in_package && let Some(v) = line.strip_prefix("name") {
            let v = v.trim_start().strip_prefix('=')?.trim().trim_matches('"');
            return Some(v.to_owned());
        }
    }
    None
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

fn scan_file(file: &Path, binary: &str, out: &mut Vec<Tagged>) {
    let Ok(src) = std::fs::read_to_string(file) else {
        return;
    };
    let mut claims = Vec::new();
    let mut defects = Vec::new();
    let mut ignored = false;
    let mut start = 0;
    for (i, raw) in src.lines().enumerate() {
        let line = raw.trim();
        let tag = line.strip_prefix("///").map(str::trim);
        if let Some(rest) = tag.and_then(|t| t.strip_prefix("claim:")) {
            if claims.is_empty() && defects.is_empty() {
                start = i + 1;
            }
            claims.extend(ids(rest));
            continue;
        }
        if let Some(rest) = tag.and_then(|t| t.strip_prefix("defect:")) {
            if claims.is_empty() && defects.is_empty() {
                start = i + 1;
            }
            defects.extend(ids(rest));
            continue;
        }
        if claims.is_empty() && defects.is_empty() {
            continue;
        }
        if line.starts_with("#[ignore") {
            ignored = true;
        }
        if let Some(name) = fn_name(line) {
            out.push(Tagged {
                name,
                binary: binary.to_owned(),
                file: file.to_owned(),
                line: start,
                ignored,
                claims: std::mem::take(&mut claims),
                defects: std::mem::take(&mut defects),
            });
            ignored = false;
        }
    }
}

fn ids(list: &str) -> impl Iterator<Item = String> + '_ {
    list.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

fn fn_name(line: &str) -> Option<String> {
    let at = line.find("fn ")?;
    let before = &line[..at];
    if !before
        .split_whitespace()
        .all(|w| matches!(w, "pub" | "async" | "pub(crate)" | "unsafe"))
    {
        return None;
    }
    let rest = &line[at + 3..];
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    (end > 0).then(|| rest[..end].to_owned())
}

/// A nextest filterset selecting exactly the tagged tests.
fn filter_expr(tagged: &[Tagged]) -> String {
    let mut parts: Vec<String> = tagged
        .iter()
        .map(|t| format!("(binary_id({}) & test(/(^|::){}$/))", t.binary, t.name))
        .collect();
    // Live cells are listed only under VERIFY_LIVE=1 (they spend money), so they join the run only
    // then.
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") {
        parts.extend(LIVE_BINARIES.iter().map(|b| format!("binary_id({b})")));
    }
    if parts.is_empty() {
        return "none()".to_owned();
    }
    parts.join(" | ")
}

/// The nextest binaries holding live cells: real clients (`crates/verify/tests/live.rs`), billing
/// reconciliation against the providers' usage reports (`tests/reconcile_live.rs`),
/// differential parity against direct provider calls (`tests/parity_live.rs`), and the catalog
/// sweep (`tests/catalog_live.rs`). Every test in them is named `CLAIMS::client::...`.
const LIVE_BINARIES: &[&str] = &[
    "beyond-ai-verify::live",
    "beyond-ai-verify::reconcile_live",
    "beyond-ai-verify::parity_live",
    "beyond-ai-verify::catalog_live",
];

/// One live cell's result, parsed from its name `CLAIMS::client::route::probe`.
struct LiveCell<'a> {
    claims: Vec<&'a str>,
    client: &'a str,
    name: &'a str,
    outcome: Outcome,
}

fn live_cells(results: &Results) -> Vec<LiveCell<'_>> {
    results
        .iter()
        .filter(|((class, _), _)| LIVE_BINARIES.contains(&class.as_str()))
        .filter_map(|((_, name), outcome)| {
            let mut parts = name.split("::");
            let claims = parts.next()?.split('+').collect();
            let client = parts.next()?;
            Some(LiveCell {
                claims,
                client,
                name,
                outcome: *outcome,
            })
        })
        .collect()
}

/// `(classname, test name) → outcome` from a nextest JUnit report. Retries (`flakyFailure`) count
/// as the final attempt's outcome.
fn parse_junit(xml: &str) -> Results {
    let mut out = BTreeMap::new();
    let mut rest = xml;
    while let Some(at) = rest.find("<testcase ") {
        rest = &rest[at..];
        let head_end = rest.find('>').unwrap_or(rest.len());
        let head = &rest[..head_end];
        let self_closing = head.ends_with('/');
        let body_end = if self_closing {
            head_end
        } else {
            rest.find("</testcase>").unwrap_or(rest.len())
        };
        let body = &rest[head_end..body_end];
        let outcome = if body.contains("<failure") || body.contains("<error") {
            Outcome::Failed
        } else if body.contains("<skipped") {
            Outcome::Skipped
        } else {
            Outcome::Passed
        };
        if let (Some(name), Some(class)) = (attr(head, "name"), attr(head, "classname")) {
            out.insert((unescape(&class), unescape(&name)), outcome);
        }
        rest = &rest[body_end.max(1)..];
    }
    out
}

fn attr(head: &str, key: &str) -> Option<String> {
    let pat = format!(" {key}=\"");
    let at = head.find(&pat)? + pat.len();
    let end = head[at..].find('"')?;
    Some(head[at..at + end].to_owned())
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn outcome_of(t: &Tagged, results: &Results) -> Option<Outcome> {
    let suffix = format!("::{}", t.name);
    results
        .iter()
        .find(|((class, name), _)| {
            class == &t.binary && (name == &t.name || name.ends_with(&suffix))
        })
        .map(|(_, o)| *o)
}

fn junit_arg(args: &[String], root: &Path) -> Result<Option<Results>, String> {
    let Some(i) = args.iter().position(|a| a == "--junit") else {
        return Ok(None);
    };
    let path = args.get(i + 1).ok_or("--junit needs a path")?;
    let p = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        root.join(path)
    };
    match std::fs::read_to_string(&p) {
        Ok(xml) => Ok(Some(parse_junit(&xml))),
        Err(e) => Err(format!("reading {}: {e}", p.display())),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Status {
    Red,
    Untested,
    Partial,
    Proven,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Self::Red => "RED",
            Self::Untested => "UNTESTED",
            Self::Partial => "PARTIAL",
            Self::Proven => "PROVEN",
        }
    }
}

struct ClaimStatus<'a> {
    claim: &'a Claim,
    status: Status,
    why: String,
    tests: Vec<&'a Tagged>,
}

fn claim_statuses<'a>(ctx: &'a Ctx, results: Option<&Results>) -> Vec<ClaimStatus<'a>> {
    let defects: BTreeMap<&str, &Defect> = ctx.defects.iter().map(|d| (d.id.as_str(), d)).collect();
    ctx.claims
        .iter()
        .map(|c| {
            let tests: Vec<&Tagged> = ctx
                .tagged
                .iter()
                .filter(|t| t.claims.iter().any(|x| x == &c.id))
                .collect();
            let open: Vec<&str> = c
                .defects
                .iter()
                .filter_map(|d| defects.get(d.as_str()))
                .filter(|d| matches!(d.state.as_str(), "suspected" | "reproduced"))
                .map(|d| d.id.as_str())
                .collect();
            let live: Vec<LiveCell<'_>> = results
                .map(live_cells)
                .unwrap_or_default()
                .into_iter()
                .filter(|cell| cell.claims.contains(&c.id.as_str()))
                .collect();
            let needs_live = c.layer != "hermetic";
            let needs_hermetic = c.layer != "live";
            let (status, why) = if !open.is_empty() {
                (Status::Red, format!("open defects: {}", open.join(", ")))
            } else if tests.is_empty() && live.is_empty() {
                (Status::Untested, "no test carries this id".to_owned())
            } else if let Some(bad) = live.iter().find(|cell| cell.outcome == Outcome::Failed) {
                (Status::Red, format!("live cell failing: {}", bad.name))
            } else if needs_hermetic && tests.is_empty() {
                (
                    Status::Partial,
                    format!("{} live cell(s); no hermetic test", live.len()),
                )
            } else {
                // Tests that reproduce an already-handled defect don't stand for the claim itself.
                let mut failed = Vec::new();
                let mut unrun = Vec::new();
                for t in &tests {
                    match results.and_then(|r| outcome_of(t, r)) {
                        Some(Outcome::Passed) => {}
                        Some(Outcome::Failed) => failed.push(t.name.as_str()),
                        Some(Outcome::Skipped) | None => unrun.push(t.name.as_str()),
                    }
                }
                if !failed.is_empty() {
                    (Status::Red, format!("failing: {}", failed.join(", ")))
                } else if !unrun.is_empty() {
                    (Status::Partial, format!("not run: {}", unrun.join(", ")))
                } else if needs_live {
                    // Every required client must have a passing cell; a claim that names no
                    // clients needs at least one.
                    let passed: Vec<&str> = live
                        .iter()
                        .filter(|cell| cell.outcome == Outcome::Passed)
                        .map(|cell| cell.client)
                        .collect();
                    let missing: Vec<&str> = c
                        .clients
                        .iter()
                        .map(String::as_str)
                        .filter(|client| !passed.contains(client))
                        .collect();
                    if passed.is_empty() {
                        (Status::Partial, "no passing live cell".to_owned())
                    } else if !missing.is_empty() {
                        (
                            Status::Partial,
                            format!("live cells missing for: {}", missing.join(", ")),
                        )
                    } else {
                        (
                            Status::Proven,
                            format!(
                                "{} test(s), {} live cell(s) pass",
                                tests.len(),
                                passed.len()
                            ),
                        )
                    }
                } else {
                    (Status::Proven, format!("{} test(s) pass", tests.len()))
                }
            };
            ClaimStatus {
                claim: c,
                status,
                why,
                tests,
            }
        })
        .collect()
}

fn status(ctx: &Ctx, args: &[String], root: &Path) -> bool {
    let results = match junit_arg(args, root) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("verify: {e}");
            return false;
        }
    };
    let rows = claim_statuses(ctx, results.as_ref());
    if args.iter().any(|a| a == "--json") {
        let claims: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.claim.id, "priority": r.claim.priority, "group": r.claim.group,
                    "text": r.claim.text, "layer": r.claim.layer, "clients": r.claim.clients,
                    "touches": r.claim.touches,
                    "status": r.status.label(), "why": r.why,
                    "tests": r.tests.iter().map(|t| format!("{}::{}", t.binary, t.name)).collect::<Vec<_>>(),
                })
            })
            .collect();
        let defects: Vec<serde_json::Value> = ctx
            .defects
            .iter()
            .map(|d| {
                serde_json::json!({"id": d.id, "severity": d.severity, "title": d.title,
                "state": d.state, "test": d.test, "fix": d.fix, "claims": d.claims})
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({"commit": head_commit(root), "claims": claims, "defects": defects})
        );
        return true;
    }

    let mut out = String::new();
    let mut group = "";
    for r in &rows {
        if r.claim.group != group {
            group = &r.claim.group;
            let _ = writeln!(out, "\n{group}");
        }
        let _ = writeln!(
            out,
            "  {:<8} {:<3} {:<9} {}  — {}",
            r.claim.id,
            r.claim.priority,
            r.status.label(),
            clip(&r.claim.text, 70),
            r.why
        );
    }
    let count = |s: Status| rows.iter().filter(|r| r.status == s).count();
    let state = |s: &str| ctx.defects.iter().filter(|d| d.state == s).count();
    let p0: Vec<&str> = rows
        .iter()
        .filter(|r| r.claim.priority == "P0" && r.status != Status::Proven)
        .map(|r| r.claim.id.as_str())
        .collect();
    let _ = writeln!(
        out,
        "\ncommit {}\nclaims {} · PROVEN {} · PARTIAL {} · RED {} · UNTESTED {}",
        head_commit(root),
        rows.len(),
        count(Status::Proven),
        count(Status::Partial),
        count(Status::Red),
        count(Status::Untested)
    );
    let _ = writeln!(
        out,
        "defects {} · suspected {} · reproduced {} · fixed {} · refuted {} · accepted {}",
        ctx.defects.len(),
        state("suspected"),
        state("reproduced"),
        state("fixed"),
        state("refuted"),
        state("accepted")
    );
    let _ = writeln!(out, "P0 not proven ({}): {}", p0.len(), p0.join(" "));
    if results.is_none() {
        let _ = writeln!(
            out,
            "(no --junit report: test outcomes unknown; run `mise run verify:status`)"
        );
    }
    print!("{out}");
    true
}

fn gate(ctx: &Ctx, args: &[String], root: &Path) -> bool {
    let results = match junit_arg(args, root) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("verify: {e}");
            return false;
        }
    };
    let mut errors: Vec<String> = Vec::new();
    let claim_ids: BTreeSet<&str> = ctx.claims.iter().map(|c| c.id.as_str()).collect();
    let defect_ids: BTreeSet<&str> = ctx.defects.iter().map(|d| d.id.as_str()).collect();

    if claim_ids.len() != ctx.claims.len() {
        errors.push("duplicate claim ids in claims.toml".to_owned());
    }
    if defect_ids.len() != ctx.defects.len() {
        errors.push("duplicate defect ids in defects.toml".to_owned());
    }
    for c in &ctx.claims {
        if !matches!(c.priority.as_str(), "P0" | "P1" | "P2") {
            errors.push(format!(
                "{}: priority {:?} is not P0/P1/P2",
                c.id, c.priority
            ));
        }
        if !matches!(c.layer.as_str(), "hermetic" | "live" | "both") {
            errors.push(format!(
                "{}: layer {:?} is not hermetic/live/both",
                c.id, c.layer
            ));
        }
        if c.touches.is_empty() {
            errors.push(format!(
                "{}: no `touches` paths, so no proof can go stale",
                c.id
            ));
        }
        for d in &c.defects {
            if !defect_ids.contains(d.as_str()) {
                errors.push(format!("{}: names unknown defect {d}", c.id));
            }
        }
    }
    let mut seen: BTreeMap<(&str, &str), &Tagged> = BTreeMap::new();
    for t in &ctx.tagged {
        let at = format!("{}:{}", t.file.display(), t.line);
        if let Some(prev) = seen.insert((&t.binary, &t.name), t) {
            errors.push(format!(
                "{at}: test `{}` is tagged twice in {} (also {}:{})",
                t.name,
                t.binary,
                prev.file.display(),
                prev.line
            ));
        }
        for c in &t.claims {
            if !claim_ids.contains(c.as_str()) {
                errors.push(format!("{at}: `{}` claims unknown id {c}", t.name));
            }
        }
        for d in &t.defects {
            if !defect_ids.contains(d.as_str()) {
                errors.push(format!("{at}: `{}` names unknown defect {d}", t.name));
            }
        }
        let reproduces_open = t.defects.iter().any(|d| {
            ctx.defects
                .iter()
                .any(|x| &x.id == d && x.state == "reproduced")
        });
        if t.ignored && !reproduces_open {
            errors.push(format!(
                "{at}: `{}` is #[ignore]d but reproduces no open defect; only reproductions may be ignored",
                t.name
            ));
        }
    }
    if let Some(r) = results.as_ref() {
        for cell in live_cells(r) {
            for c in &cell.claims {
                if !claim_ids.contains(c) {
                    errors.push(format!("live cell `{}` claims unknown id {c}", cell.name));
                }
            }
        }
    }
    for d in &ctx.defects {
        if !STATES.contains(&d.state.as_str()) {
            errors.push(format!(
                "{}: state {:?} is not one of {STATES:?}",
                d.id, d.state
            ));
            continue;
        }
        for c in &d.claims {
            if !claim_ids.contains(c.as_str()) {
                errors.push(format!("{}: names unknown claim {c}", d.id));
            }
        }
        if d.state == "suspected" {
            continue;
        }
        // A defect closed by a live cell names it as `live:<cell name>`; the cell's own outcome
        // decides, and it must have run (live cells only exist under VERIFY_LIVE=1).
        if let Some(cell) = d.test.strip_prefix("live:") {
            if d.state == "reproduced" {
                errors.push(format!(
                    "{}: a reproduction must be a hermetic #[ignore]d test, not a live cell",
                    d.id
                ));
            }
            if d.note.trim().is_empty() {
                errors.push(format!(
                    "{}: {} by a live cell needs a `note` with the evidence",
                    d.id, d.state
                ));
            }
            if let Some(r) = results.as_ref() {
                let outcome = LIVE_BINARIES
                    .iter()
                    .find_map(|b| r.get(&((*b).to_owned(), cell.to_owned())))
                    .copied();
                match outcome {
                    Some(Outcome::Passed) => {}
                    Some(Outcome::Failed) => errors.push(format!(
                        "{}: live cell `{cell}` fails but the defect is {}",
                        d.id, d.state
                    )),
                    _ if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") => {
                        errors.push(format!("{}: live cell `{cell}` did not run", d.id));
                    }
                    _ => {}
                }
            }
            continue;
        }
        let Some(t) = ctx
            .tagged
            .iter()
            .find(|t| t.name == d.test && t.defects.iter().any(|x| x == &d.id))
        else {
            errors.push(format!(
                "{} ({}): test {:?} not found with a `/// defect: {}` tag",
                d.id, d.state, d.test, d.id
            ));
            continue;
        };
        let outcome = results.as_ref().and_then(|r| outcome_of(t, r));
        match d.state.as_str() {
            "reproduced" => {
                if !t.ignored {
                    errors.push(format!(
                        "{}: reproduction `{}` must be #[ignore]d so the normal suite stays green",
                        d.id, t.name
                    ));
                }
                match outcome {
                    Some(Outcome::Failed) => {}
                    Some(Outcome::Passed) => errors.push(format!(
                        "{}: reproduction `{}` now PASSES. Mark it fixed (with the commit) or refuted, and drop #[ignore]",
                        d.id, t.name
                    )),
                    _ if results.is_some() => errors.push(format!("{}: reproduction `{}` did not run", d.id, t.name)),
                    _ => {}
                }
            }
            "fixed" | "refuted" | "accepted" => {
                if t.ignored {
                    errors.push(format!(
                        "{}: `{}` is {} but still #[ignore]d",
                        d.id, t.name, d.state
                    ));
                }
                if d.state == "fixed" && d.fix.trim().is_empty() {
                    errors.push(format!("{}: fixed without a `fix` commit", d.id));
                }
                if d.state != "fixed" && d.note.trim().is_empty() {
                    errors.push(format!(
                        "{}: {} needs a `note` with the evidence or decision",
                        d.id, d.state
                    ));
                }
                match outcome {
                    Some(Outcome::Passed) | None if results.is_none() => {}
                    Some(Outcome::Passed) => {}
                    Some(Outcome::Failed) => {
                        errors.push(format!(
                            "{}: `{}` fails but the defect is {}",
                            d.id, t.name, d.state
                        ));
                    }
                    _ => errors.push(format!("{}: `{}` did not run", d.id, t.name)),
                }
            }
            _ => {}
        }
    }
    if errors.is_empty() {
        println!(
            "verify gate: ok ({} claims, {} defects, {} tagged tests{})",
            ctx.claims.len(),
            ctx.defects.len(),
            ctx.tagged.len(),
            if results.is_some() {
                ", results checked"
            } else {
                ", static only"
            }
        );
        true
    } else {
        for e in &errors {
            eprintln!("  ✗ {e}");
        }
        eprintln!("verify gate: {} problem(s)", errors.len());
        false
    }
}

fn head_commit(root: &Path) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        format!("{s:<n$}")
    } else {
        let mut t: String = s.chars().take(n - 1).collect();
        t.push('…');
        t
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn tags_attach_to_the_next_fn_and_see_ignore() {
        let dir = std::env::temp_dir().join(format!("verify-tags-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.rs");
        std::fs::write(
            &f,
            "/// claim: SEC-1, SEC-5\n/// defect: D01\n#[tokio::test]\n#[ignore = \"D01\"]\nasync fn a_test() {}\n\n#[test]\nfn untagged() {}\n/// claim: E1\n#[test]\nfn b() {}\n",
        )
        .unwrap();
        let mut out = Vec::new();
        scan_file(&f, "pkg::t", &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "a_test");
        assert_eq!(out[0].claims, ["SEC-1", "SEC-5"]);
        assert_eq!(out[0].defects, ["D01"]);
        assert!(out[0].ignored);
        assert_eq!(out[1].name, "b");
        assert!(!out[1].ignored);
    }

    #[test]
    fn junit_outcomes() {
        let xml = r#"<testsuites><testsuite name="pkg::t">
<testcase name="a_test" classname="pkg::t" time="0.1"><failure message="x"/></testcase>
<testcase name="m::tests::b" classname="pkg" time="0.1"/>
<testcase name="c" classname="pkg::t" time="0"><skipped/></testcase>
</testsuite></testsuites>"#;
        let r = parse_junit(xml);
        assert!(r[&("pkg::t".into(), "a_test".into())] == Outcome::Failed);
        assert!(r[&("pkg".into(), "m::tests::b".into())] == Outcome::Passed);
        assert!(r[&("pkg::t".into(), "c".into())] == Outcome::Skipped);
        let t = Tagged {
            name: "b".into(),
            binary: "pkg".into(),
            file: PathBuf::new(),
            line: 0,
            ignored: false,
            claims: vec![],
            defects: vec![],
        };
        assert!(outcome_of(&t, &r) == Some(Outcome::Passed));
    }
}
