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
//! ```

use rates_sync::classify::{self, Class, Report};
use rates_sync::{
    GENERATED, Result, SOURCES, audit, build, diff, emit, fetch, invoice, repo_root, snapshot,
    sources,
};
use std::path::Path;
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
    let r = match cmd {
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
            "usage: rates-sync sync | generate | check | drift | classify [--apply] | rate-version | audit"
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
