//! Fetching sources and writing snapshots.

use crate::Result;
use crate::snapshot::{self, Entry, sha256_hex};
use crate::sources::{self, Auth, SourceDef};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

pub fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent("beyond-ai-rates-sync/0.1 (+https://github.com/beyondoss/ai)")
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|e| format!("HTTP client: {e}"))
}

/// GET `url`, with up to six attempts on a 429, a 5xx or a transport error, honouring a
/// `retry-after` (seconds) up to two minutes.
pub fn get(
    client: &reqwest::blocking::Client,
    url: &str,
    headers: &[(&str, String)],
) -> Result<Vec<u8>> {
    let mut last = String::new();
    let mut wait = Duration::ZERO;
    for attempt in 0..6u32 {
        if attempt > 0 {
            std::thread::sleep(wait.max(Duration::from_secs(2u64.pow(attempt))));
        }
        let mut req = client.get(url);
        for (k, v) in headers {
            req = req.header(*k, v);
        }
        match req.send() {
            Ok(r) => {
                let status = r.status();
                wait = r
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .map_or(Duration::ZERO, |s| Duration::from_secs(s.min(120)));
                let body = r.bytes().map_err(|e| format!("{url}: {e}"))?;
                if status.is_success() {
                    return Ok(body.to_vec());
                }
                last = format!(
                    "{url}: HTTP {status}: {}",
                    String::from_utf8_lossy(&body[..body.len().min(300)])
                );
                if !(status.as_u16() == 429 || status.is_server_error()) {
                    return Err(last);
                }
            }
            Err(e) => last = format!("{url}: {e}"),
        }
    }
    Err(last)
}

/// The outcome of fetching one source.
pub enum Fetched {
    /// Fetched and normalized: (raw sha256, snapshot text).
    Got(String, String),
    /// Its key is not set.
    NoKey(&'static str),
}

pub fn fetch_one(
    client: &reqwest::blocking::Client,
    d: &SourceDef,
    or_slugs: &[String],
) -> Result<Fetched> {
    let mut headers = Vec::new();
    if let Auth::Bearer(var) = d.auth {
        match std::env::var(var) {
            Ok(k) if !k.is_empty() => headers.push(("authorization", format!("Bearer {k}"))),
            _ => return Ok(Fetched::NoKey(var)),
        }
    }
    let raw = get(client, &d.url, &headers)?;
    let text = sources::normalize(d, &raw, or_slugs)?;
    Ok(Fetched::Got(sha256_hex(&raw), text))
}

/// Fetch every source into `dir`, keeping each entry's `fetched` date when its snapshot is
/// unchanged. A keyed source whose key is unset keeps its committed snapshot when `allow_missing`
/// (the drift job without that secret) and is an error otherwise. Returns the ids whose snapshot
/// changed, and the ids skipped for a missing key.
pub fn fetch_all(
    dir: &Path,
    defs: &[SourceDef],
    or_slugs: &[String],
    today: &str,
    allow_missing: bool,
) -> Result<(Vec<String>, Vec<String>)> {
    let client = client()?;
    let mut entries: BTreeMap<String, Entry> = snapshot::read_manifest(dir).unwrap_or_default();
    entries.retain(|id, _| defs.iter().any(|d| &d.id == id));
    let mut changed = Vec::new();
    let mut skipped = Vec::new();
    let mut errors = Vec::new();
    for d in defs {
        match fetch_one(&client, d, or_slugs) {
            Ok(Fetched::Got(raw_sha, text)) => {
                let sha = sha256_hex(text.as_bytes());
                let same = entries
                    .get(&d.id)
                    .is_some_and(|e| e.sha256 == sha && e.url == d.url && e.file == d.file);
                if !same {
                    let path = dir.join(&d.file);
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| format!("{}: {e}", parent.display()))?;
                    }
                    std::fs::write(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
                    changed.push(d.id.clone());
                    entries.insert(
                        d.id.clone(),
                        Entry {
                            id: d.id.clone(),
                            url: d.url.clone(),
                            file: d.file.clone(),
                            fetched: today.to_owned(),
                            raw_sha256: raw_sha,
                            sha256: sha,
                        },
                    );
                }
            }
            Ok(Fetched::NoKey(var)) => {
                if allow_missing && entries.contains_key(&d.id) {
                    skipped.push(format!("{} (no {var})", d.id));
                } else {
                    errors.push(format!("{}: {var} is not set", d.id));
                }
            }
            Err(e) => errors.push(e),
        }
    }
    // Remove snapshot files no source names any more.
    let keep: Vec<String> = entries.values().map(|e| e.file.clone()).collect();
    remove_strays(dir, dir, &keep)?;
    snapshot::write_manifest(dir, &entries)?;
    if errors.is_empty() {
        Ok((changed, skipped))
    } else {
        Err(errors.join("\n"))
    }
}

fn remove_strays(root: &Path, dir: &Path, keep: &[String]) -> Result<()> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            remove_strays(root, &p, keep)?;
            if std::fs::read_dir(&p)
                .map(|mut r| r.next().is_none())
                .unwrap_or(false)
            {
                let _ = std::fs::remove_dir(&p);
            }
        } else if let Ok(rel) = p.strip_prefix(root) {
            let rel = rel.to_string_lossy().replace('\\', "/");
            if rel != "manifest.toml" && !keep.contains(&rel) {
                std::fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            }
        }
    }
    Ok(())
}

/// Today's UTC date, `YYYY-MM-DD`.
pub fn today() -> String {
    date_of(i64::try_from(now()).unwrap_or(0))
}

/// Unix seconds now.
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The UTC date of a Unix second, `YYYY-MM-DD`.
pub fn date_of(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    // Civil from days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}
