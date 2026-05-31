//! Diagnostics (PATTERNS.md `doctor` pattern): fast prerequisite checks, exit 0/1.

use crate::config::AiConfig;

pub struct CheckResult {
    pub name: &'static str,
    pub passed: bool,
    pub message: String,
    pub hint: Option<String>,
}

fn pass(name: &'static str, message: impl Into<String>) -> CheckResult {
    CheckResult {
        name,
        passed: true,
        message: message.into(),
        hint: None,
    }
}

fn fail(name: &'static str, message: impl Into<String>, hint: &str) -> CheckResult {
    CheckResult {
        name,
        passed: false,
        message: message.into(),
        hint: Some(hint.to_string()),
    }
}

pub async fn run_checks(config: &AiConfig) -> Vec<CheckResult> {
    let mut out = Vec::new();

    // NATS / slipstream reachability — without it we can't load signing keys or the deny-set.
    match store::nats_connect(
        &config.nats_url,
        config.nats_creds.as_ref().map(|s| s.expose()),
        config.nats_creds_file.as_deref(),
    )
    .await
    {
        Ok(_) => out.push(pass("nats", format!("connected to {}", config.nats_url))),
        Err(e) => out.push(fail(
            "nats",
            e.to_string(),
            "check AI_NATS_URL and credentials",
        )),
    }

    out
}

pub fn print_results(title: &str, results: &[CheckResult]) {
    println!("== {title} ==");
    for r in results {
        let mark = if r.passed { "ok" } else { "FAIL" };
        println!("[{mark}] {}: {}", r.name, r.message);
        if let (false, Some(hint)) = (r.passed, &r.hint) {
            println!("       hint: {hint}");
        }
    }
}
