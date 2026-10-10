//! `rates-sync`: see the library docs, and `crates/providers/ARCHITECTURE.md`.
//!
//! ```text
//! rates-sync sync      fetch every source, write the snapshots, regenerate the table
//! rates-sync generate  regenerate the table from the committed snapshots (offline)
//! rates-sync check     fail if the committed table is not what the snapshots generate (offline)
//! rates-sync drift     re-fetch into a scratch copy; fail, listing the diff, if any price moved
//! rates-sync classify  re-fetch into a scratch copy and print, as JSON, whether the change is
//!                      none (exit 0), routine (2), review (3) or broken (4); `--apply` writes a
//!                      routine or review change into the snapshots and the table
//! rates-sync rate-version  set RATE_VERSION (and the golden vectors' rate_version) to name the
//!                      table this binary was compiled with
//! rates-sync audit     LiteLLM and models.dev as second opinions, plus the vendors' invoices
//! rates-sync catalog-drift [--save DIR | --inputs DIR]
//!                      fetch every listing, deprecation page and rate source into DIR (default a
//!                      scratch directory), or replay a saved DIR, and print the catalog's new
//!                      models and retirements as JSON (crates/rates-sync/src/catalog_drift.rs)
//! rates-sync catalog-apply --inputs DIR SLUG
//!                      write one successor or retirement finding of DIR into catalog.rs, the truth
//!                      file and the snapshots (then run `generate` and `rate-version`)
//! ```

use rates_sync::classify::{self, Class, Report};
use rates_sync::{
    GENERATED, Result, SOURCES, TRUTH, audit, build, catalog_drift, catalog_edit, diff, emit,
    fetch, invoice, repo_root, snapshot, sources,
};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map_or("", String::as_str);
    let root = repo_root();
    if cmd == "classify" {
        let apply = args.get(1).is_some_and(|a| a == "--apply");
        return match run_classify(&root, apply) {
            Ok(class) => ExitCode::from(class.exit_code()),
            Err(e) => {
                eprintln!("rates-sync classify: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from)
    };
    let r = match cmd {
        "catalog-drift" => drift_catalog(&root, flag("--save"), flag("--inputs")),
        "catalog-apply" => match (flag("--inputs"), args.last()) {
            (Some(dir), Some(slug)) if args.len() == 4 => apply_catalog(&root, &dir, slug),
            _ => Err("usage: rates-sync catalog-apply --inputs DIR SLUG".into()),
        },
        "sync" => sync(&root),
        "rate-version" => rate_version(&root),
        "generate" => write_generated(&root),
        "check" => check(&root),
        "drift" => drift(&root),
        "audit" => run_audit(&root),
        "dump" => {
            print!("{}", rates_sync::canon::dump());
            Ok(())
        }
        _ => Err(
            "usage: rates-sync sync | generate | check | drift | classify [--apply] | rate-version | audit | catalog-drift [--save DIR | --inputs DIR] | catalog-apply --inputs DIR SLUG"
                .into(),
        ),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rates-sync {cmd}: {e}");
            ExitCode::FAILURE
        }
    }
}

fn write_generated(root: &Path) -> Result<()> {
    let text = rates_sync::generate(root, &root.join(SOURCES))?;
    let path = root.join(GENERATED);
    let old = std::fs::read_to_string(&path).unwrap_or_default();
    if old == text {
        println!("{GENERATED}: unchanged");
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
    if let Some(d) = diff::lines(&old, &text) {
        println!("{GENERATED}: regenerated\n{d}");
    }
    println!(
        "Rates changed: run `cargo run -p beyond-ai-rates-sync -- rate-version` for the new RATE_VERSION\n\
         (`mise run rates:sync` does), and update verify/pricing_vectors.json for any vector whose rates moved."
    );
    Ok(())
}

fn sync(root: &Path) -> Result<()> {
    let spec = rates_sync::read_spec(root)?;
    let slugs = spec.openrouter_slugs();
    let defs = sources::all(&slugs);
    let dir = root.join(SOURCES);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let (changed, _) = fetch::fetch_all(&dir, &defs, &slugs, &fetch::today(), false)?;
    if changed.is_empty() {
        println!("sources: unchanged");
    } else {
        println!("sources changed:\n  {}", changed.join("\n  "));
    }
    write_generated(root)
}

fn check(root: &Path) -> Result<()> {
    let text = rates_sync::generate(root, &root.join(SOURCES))?;
    let old = std::fs::read_to_string(root.join(GENERATED)).unwrap_or_default();
    match diff::lines(&old, &text) {
        None => {
            println!("{GENERATED} is current");
            Ok(())
        }
        Some(d) => Err(format!(
            "{GENERATED} is not what the snapshots generate (run `mise run rates:generate`):\n{d}"
        )),
    }
}

/// Copy the committed store to a scratch directory, re-fetch into it, regenerate, and compare.
fn drift(root: &Path) -> Result<()> {
    let spec = rates_sync::read_spec(root)?;
    let slugs = spec.openrouter_slugs();
    let defs = sources::all(&slugs);
    let scratch = scratch_copy(&root.join(SOURCES))?;
    let fetched = fetch::fetch_all(&scratch, &defs, &slugs, &fetch::today(), true);
    let report = (|| -> Result<Option<String>> {
        let (changed, skipped) = fetched?;
        for s in &skipped {
            println!("::warning::not re-fetched (its key is not set): {s}");
        }
        if !changed.is_empty() {
            println!(
                "sources whose snapshot changed:\n  {}",
                changed.join("\n  ")
            );
        }
        let new = rates_sync::generate(root, &scratch)
            .map_err(|e| format!("a source no longer generates the table: {e}"))?;
        let old = std::fs::read_to_string(root.join(GENERATED)).map_err(|e| e.to_string())?;
        Ok(diff::lines(&old, &new))
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    match report? {
        None => {
            println!("no vendor price changed");
            Ok(())
        }
        Some(d) => Err(format!(
            "vendor prices changed; run `mise run rates:sync`, review, and bump RATE_VERSION:\n{d}"
        )),
    }
}

/// A fresh copy of the committed store to re-fetch into.
fn scratch_copy(committed: &Path) -> Result<std::path::PathBuf> {
    let scratch = std::env::temp_dir().join(format!("rates-drift-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    copy_dir(committed, &scratch)?;
    Ok(scratch)
}

/// Re-fetch into a scratch copy and classify what moved (`classify.rs`), printing the report as
/// JSON on stdout. With `apply`, a routine or review change replaces the committed snapshots and
/// regenerates the table; a broken one writes nothing.
fn run_classify(root: &Path, apply: bool) -> Result<Class> {
    let spec = rates_sync::read_spec(root)?;
    let slugs = spec.openrouter_slugs();
    let defs = sources::all(&slugs);
    let committed = root.join(SOURCES);
    let old = build::build(&snapshot::load(&committed, &defs)?, &spec)
        .map_err(|e| format!("the committed snapshots do not generate the table: {e}"))?;
    let scratch = scratch_copy(&committed)?;
    let r = (|| -> Result<Class> {
        let (report, changed, skipped) =
            match fetch::fetch_all(&scratch, &defs, &slugs, &fetch::today(), true) {
                Err(e) => (Report::broken("fetch", e), Vec::new(), Vec::new()),
                Ok((changed, skipped)) => {
                    let new = snapshot::load(&scratch, &defs)
                        .and_then(|store| build::build(&store, &spec))
                        .and_then(|t| emit::render(&t).map(|text| (t, text)));
                    let entries = snapshot::read_manifest(&scratch)?;
                    let report = classify::classify(
                        &old,
                        new.as_ref().map(|(t, _)| t).map_err(String::as_str),
                        &entries,
                    );
                    if apply
                        && matches!(report.class, Class::Routine | Class::Review)
                        && let Ok((_, text)) = &new
                    {
                        std::fs::remove_dir_all(&committed)
                            .map_err(|e| format!("{}: {e}", committed.display()))?;
                        copy_dir(&scratch, &committed)?;
                        let path = root.join(GENERATED);
                        std::fs::write(&path, text)
                            .map_err(|e| format!("{}: {e}", path.display()))?;
                    }
                    (report, changed, skipped)
                }
            };
        eprintln!(
            "rates-sync classify: {}\n{}",
            report.class.as_str(),
            report.markdown(&changed, &skipped)
        );
        println!("{}", report.to_json(&changed, &skipped));
        Ok(report.class)
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    r
}

/// Point `RATE_VERSION` (and the golden vectors' `rate_version`) at the table this binary was
/// compiled with: `{today}.{hash}`, or unchanged when the hash already matches. Deterministic for
/// a table and a day. Run it in a fresh `cargo run` after regenerating, so the linked providers
/// crate is the new table.
fn rate_version(root: &Path) -> Result<()> {
    let hash = providers::rates::table_hash();
    let cur = providers::rates::RATE_VERSION;
    let v = match cur.split_once('.') {
        Some((_, h)) if h == hash => cur.to_owned(),
        _ => format!("{}.{hash}", fetch::today()),
    };
    let rewrite = |rel: &str, from: &str, to: &str| -> Result<()> {
        let path = root.join(rel);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{rel}: {e}"))?;
        if text.matches(from).count() != 1 {
            return Err(format!("{rel}: expected exactly one {from:?}"));
        }
        std::fs::write(&path, text.replacen(from, to, 1)).map_err(|e| format!("{rel}: {e}"))
    };
    let decl = |v: &str| format!("pub const RATE_VERSION: &str = \"{v}\";");
    rewrite("crates/providers/src/rates.rs", &decl(cur), &decl(&v))?;
    let vectors = "verify/pricing_vectors.json";
    let doc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join(vectors)).map_err(|e| format!("{vectors}: {e}"))?,
    )
    .map_err(|e| format!("{vectors}: {e}"))?;
    let had = doc["rate_version"]
        .as_str()
        .ok_or_else(|| format!("{vectors}: no rate_version"))?;
    let field = |v: &str| format!("\"rate_version\": \"{v}\"");
    rewrite(vectors, &field(had), &field(&v))?;
    println!("{v}");
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for e in std::fs::read_dir(from)
        .map_err(|e| format!("{}: {e}", from.display()))?
        .flatten()
    {
        let p = e.path();
        let dest = to.join(e.file_name());
        if p.is_dir() {
            copy_dir(&p, &dest)?;
        } else {
            std::fs::copy(&p, &dest).map_err(|e| format!("{}: {e}", p.display()))?;
        }
    }
    Ok(())
}

fn run_audit(root: &Path) -> Result<()> {
    let spec = rates_sync::read_spec(root)?;
    let defs = sources::all(&spec.openrouter_slugs());
    let store = snapshot::load(&root.join(SOURCES), &defs)?;
    let table = build::build(&store, &spec)?;
    let c = fetch::client()?;
    let json = |url: &str| -> Result<serde_json::Value> {
        serde_json::from_slice(&fetch::get(&c, url, &[])?).map_err(|e| format!("{url}: {e}"))
    };
    let lines = audit::run(&table, &json(audit::LITELLM)?, &json(audit::MODELS_DEV)?)?;
    println!(
        "== third-party disagreements (not authoritative; check each against the primary source)"
    );
    for l in &lines {
        println!("{l}");
    }
    let mut bad = 0;
    let now = fetch::now();
    for (vendor, var) in [
        ("openai", "OPENAI_ADMIN_KEY"),
        ("anthropic", "ANTHROPIC_ADMIN_KEY"),
    ] {
        let Ok(key) = std::env::var(var) else {
            println!("== {vendor} invoices: skipped ({var} is not set)");
            continue;
        };
        let checks = if vendor == "openai" {
            invoice::openai(&table, &key, 30, now)
        } else {
            invoice::anthropic(&table, &key, 30, now)
        };
        match checks {
            Ok(mut checks) => {
                invoice::accept_superseded(&mut checks, &spec)?;
                let n = checks.len();
                println!("== {vendor} invoices: {n} (day, model, token kind) lines compared");
                // RATES_AUDIT_VERBOSE=1 lists the lines that agree too.
                let verbose = std::env::var("RATES_AUDIT_VERBOSE").is_ok_and(|v| v == "1");
                for c in checks.iter().filter(|c| verbose || !c.ok) {
                    if !c.ok {
                        bad += 1;
                    }
                    println!(
                        "{} {} {} {} {}: {} tokens billed ${:.9}, the card gives ${:.9}",
                        match (c.ok, c.superseded) {
                            (true, false) => "ok",
                            (true, true) => "ok (superseded rate)",
                            _ => "MISMATCH",
                        },
                        c.vendor,
                        c.day,
                        c.model,
                        c.kind,
                        c.tokens,
                        c.billed,
                        c.expected
                    );
                }
            }
            Err(e) => println!("== {vendor} invoices: error: {e}"),
        }
    }
    if bad > 0 {
        return Err(format!("{bad} invoice lines disagree with the table"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// catalog-drift
// ---------------------------------------------------------------------------------------------

const CATALOG: &str = "crates/providers/src/catalog.rs";

/// Fetch everything `catalog_drift::detect` reads into `dir`: the rate sources (a re-fetched copy
/// of the committed store), OpenRouter's whole model list, the vendors' own listings (where the
/// key is set), the deprecation pages, and the endpoint lists of the slugs new models would be
/// priced from. A source that can't be fetched keeps its committed snapshot, or is left out.
fn fetch_inputs(root: &Path, dir: &Path) -> Result<()> {
    let spec = rates_sync::read_spec(root)?;
    let slugs = spec.openrouter_slugs();
    let defs = sources::all(&slugs);
    let _ = std::fs::remove_dir_all(dir);
    let store = catalog_drift::sources_dir(dir);
    copy_dir(&root.join(SOURCES), &store)?;
    let today = fetch::today();
    match fetch::fetch_all(&store, &defs, &slugs, &today, true) {
        Ok((_, skipped)) => {
            for s in skipped {
                println!("::warning::not re-fetched (its key is not set): {s}");
            }
        }
        Err(e) => println!("::warning::some rate sources kept their committed snapshot: {e}"),
    }
    let c = fetch::client()?;
    let save = |rel: &str, bytes: &[u8]| -> Result<()> {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))
    };
    save("today", format!("{today}\n").as_bytes())?;
    save(
        "openrouter-models.json",
        &fetch::get(&c, sources::OPENROUTER_MODELS, &[])?,
    )?;
    let key = |var: &str| std::env::var(var).ok().filter(|k| !k.is_empty());
    let keyed = [
        (
            "anthropic-models.json",
            catalog_drift::ANTHROPIC_MODELS,
            "ANTHROPIC_API_KEY",
            key("ANTHROPIC_API_KEY").map(|k| {
                vec![
                    ("x-api-key", k),
                    ("anthropic-version", "2023-06-01".to_owned()),
                ]
            }),
        ),
        (
            "openai-models.json",
            catalog_drift::OPENAI_MODELS,
            "OPENAI_API_KEY",
            key("OPENAI_API_KEY").map(|k| vec![("authorization", format!("Bearer {k}"))]),
        ),
        (
            "bedrock-profiles.json",
            catalog_drift::BEDROCK_PROFILES,
            "AWS_BEARER_TOKEN_BEDROCK",
            key("AWS_BEARER_TOKEN_BEDROCK").map(|k| vec![("authorization", format!("Bearer {k}"))]),
        ),
    ];
    for (file, url, var, headers) in keyed {
        match headers {
            None => println!("::warning::{var} is not set: {url} not read"),
            Some(h) => match fetch::get(&c, url, &h) {
                Ok(b) => save(file, &b)?,
                Err(e) => println!("::warning::{url}: {e}"),
            },
        }
    }
    for (p, url) in catalog_drift::DEPRECATION_PAGES {
        match fetch::get(&c, url, &[]) {
            Ok(b) => save(&format!("deprecations/{p}.md"), &b)?,
            Err(e) => println!("::warning::{url}: {e}"),
        }
    }
    let truth = read_truth(root)?;
    let inputs = catalog_drift::Inputs::load(dir, &defs)?;
    let cat = catalog_drift::Catalog {
        rows: providers::catalog::MODEL_ROUTES,
        spec: &spec,
        truth: &truth,
    };
    for slug in catalog_drift::wanted_endpoints(&inputs, &cat)? {
        let url = format!("https://openrouter.ai/api/v1/models/{slug}/endpoints");
        match fetch::get(&c, &url, &[]) {
            Ok(b) => save(&format!("endpoints/{slug}.json"), &b)?,
            Err(e) => println!("::warning::{url}: {e}"),
        }
    }
    Ok(())
}

fn read_truth(root: &Path) -> Result<toml::Table> {
    let text = std::fs::read_to_string(root.join(TRUTH)).map_err(|e| format!("{TRUTH}: {e}"))?;
    text.parse().map_err(|e| format!("{TRUTH}: {e}"))
}

fn detect_saved(
    root: &Path,
    dir: &Path,
) -> Result<(catalog_drift::Inputs, Vec<catalog_drift::Finding>)> {
    let spec = rates_sync::read_spec(root)?;
    let truth = read_truth(root)?;
    let inputs = catalog_drift::Inputs::load(dir, &sources::all(&spec.openrouter_slugs()))?;
    let cat = catalog_drift::Catalog {
        rows: providers::catalog::MODEL_ROUTES,
        spec: &spec,
        truth: &truth,
    };
    let findings = catalog_drift::detect(&inputs, &cat)?;
    Ok((inputs, findings))
}

/// Fetch (or replay) the inputs and print every finding, and the needs-human issue body, as JSON.
fn drift_catalog(root: &Path, save: Option<PathBuf>, inputs: Option<PathBuf>) -> Result<()> {
    let dir = match (inputs, save) {
        (Some(d), _) => d,
        (None, save) => {
            let d = save.unwrap_or_else(|| {
                std::env::temp_dir().join(format!("catalog-drift-{}", std::process::id()))
            });
            fetch_inputs(root, &d)?;
            eprintln!("catalog-drift: inputs saved in {}", d.display());
            d
        }
    };
    let (_, findings) = detect_saved(root, &dir)?;
    for f in &findings {
        eprintln!("catalog-drift: {} {}", f.class.as_str(), f.slug);
    }
    let out = serde_json::json!({
        "findings": findings.iter().map(catalog_drift::Finding::to_json).collect::<Vec<_>>(),
        "issue": catalog_drift::issue_markdown(&findings),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
    );
    Ok(())
}

/// Apply one successor or retirement finding of a saved run to the tree: the catalog row, the
/// truth file, and the snapshots its cards and OpenRouter slugs read (taken from the run, never
/// re-fetched). `generate` and `rate-version`, run after, rebuild against the edited catalog.
fn apply_catalog(root: &Path, dir: &Path, slug: &str) -> Result<()> {
    let (inputs, findings) = detect_saved(root, dir)?;
    let f = findings
        .iter()
        .find(|f| f.slug == slug)
        .ok_or_else(|| format!("no finding {slug} in {}", dir.display()))?;
    let edit = f
        .edit
        .as_ref()
        .ok_or_else(|| format!("{slug} is {}: no rule edits it", f.class.as_str()))?;
    let rewrite = |rel: &str, g: &dyn Fn(&str) -> Result<String>| -> Result<()> {
        let p = root.join(rel);
        let old = std::fs::read_to_string(&p).map_err(|e| format!("{rel}: {e}"))?;
        std::fs::write(&p, g(&old)?).map_err(|e| format!("{rel}: {e}"))
    };
    rewrite(CATALOG, &|t| catalog_edit::apply_catalog(t, edit))?;
    rewrite(TRUTH, &|t| catalog_edit::apply_truth(t, edit))?;

    // The snapshots: OpenRouter's model list for the new slug set, a new slug's endpoints, and
    // (a successor) the vendor pages its cards read, as this run fetched them.
    let spec = rates_sync::read_spec(root)?;
    let slugs = spec.openrouter_slugs();
    let defs = sources::all(&slugs);
    let committed = root.join(SOURCES);
    let mut entries = snapshot::read_manifest(&committed)?;
    let refresh: Vec<String> = match edit {
        catalog_edit::Edit::Add(a) => a.refresh.clone(),
        catalog_edit::Edit::Retire(_) => vec!["openrouter.models".into()],
    };
    for d in &defs {
        let from_raw = if d.id == "openrouter.models" {
            Some(inputs.openrouter.as_str())
        } else {
            d.id.strip_prefix("openrouter.endpoints/")
                .filter(|_| !entries.contains_key(&d.id))
                .and_then(|s| inputs.endpoints.get(s))
                .map(String::as_str)
        };
        let (raw_sha, text, fetched) = match from_raw {
            Some(raw) => (
                snapshot::sha256_hex(raw.as_bytes()),
                sources::normalize(d, raw.as_bytes(), &slugs)?,
                inputs.today.clone(),
            ),
            None if refresh.contains(&d.id) => {
                let e = inputs
                    .store
                    .entries
                    .get(&d.id)
                    .ok_or_else(|| format!("{}: not in the run's snapshots", d.id))?;
                (
                    e.raw_sha256.clone(),
                    inputs.store.get(&d.id)?.to_owned(),
                    e.fetched.clone(),
                )
            }
            None => continue,
        };
        let sha = snapshot::sha256_hex(text.as_bytes());
        if entries.get(&d.id).is_some_and(|e| e.sha256 == sha) {
            continue;
        }
        let path = committed.join(&d.file);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
        entries.insert(
            d.id.clone(),
            snapshot::Entry {
                id: d.id.clone(),
                url: d.url.clone(),
                file: d.file.clone(),
                fetched,
                raw_sha256: raw_sha,
                sha256: sha,
            },
        );
    }
    // A slug no row prices any more leaves the store.
    let gone: Vec<String> = entries
        .keys()
        .filter(|id| !defs.iter().any(|d| &d.id == *id))
        .cloned()
        .collect();
    for id in gone {
        if let Some(e) = entries.remove(&id) {
            let _ = std::fs::remove_file(committed.join(&e.file));
        }
    }
    snapshot::write_manifest(&committed, &entries)?;
    println!("{}", f.model);
    Ok(())
}
