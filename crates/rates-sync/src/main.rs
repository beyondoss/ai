//! `rates-sync`: see the library docs, and `crates/providers/ARCHITECTURE.md`.
//!
//! ```text
//! rates-sync sync      fetch every source, write the snapshots, regenerate the table
//! rates-sync generate  regenerate the table from the committed snapshots (offline)
//! rates-sync check     fail if the committed table is not what the snapshots generate (offline)
//! rates-sync drift     re-fetch into a scratch copy; fail, listing the diff, if any price moved
//! rates-sync audit     LiteLLM and models.dev as second opinions, plus the vendors' invoices
//! ```

use rates_sync::{
    GENERATED, Result, SOURCES, audit, build, diff, fetch, invoice, repo_root, snapshot, sources,
};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map_or("", String::as_str);
    let root = repo_root();
    let r = match cmd {
        "sync" => sync(&root),
        "generate" => write_generated(&root),
        "check" => check(&root),
        "drift" => drift(&root),
        "audit" => run_audit(&root),
        "dump" => {
            print!("{}", rates_sync::canon::dump());
            Ok(())
        }
        _ => Err("usage: rates-sync sync | generate | check | drift | audit".into()),
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
        "Rates changed: run `cargo test -p beyond-ai-providers --test rates_truth` for the new RATE_VERSION,\n\
         and update verify/pricing_vectors.json for any vector whose rates moved."
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
    let committed = root.join(SOURCES);
    let scratch = std::env::temp_dir().join(format!("rates-drift-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    copy_dir(&committed, &scratch)?;
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
