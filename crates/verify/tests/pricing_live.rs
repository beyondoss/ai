//! Live pricing reconciliation (claims BIL-24, BIL-25): the gateway's logged `cost_micros`
//! against what each vendor actually bills.
//!
//! **BIL-24, one case per priced dimension.** Each case boots a gateway holding one provider's
//! pool key, sends one real managed request (or two) that exercises the dimension, and holds
//! the `ai.usage` row to the vendor's own report:
//!
//! - Where the vendor reports dollars, those must match exactly:
//!   - OpenRouter: the row's `upstream_cost_usd` equals `usage.cost` and the generation API's
//!     `total_cost`, and `cost_micros` is that × 1.055 (the credit fee);
//!   - xAI: the row's tokens and tools, priced by the rate table, equal xAI's
//!     `cost_in_usd_ticks` to the micro-dollar.
//! - Where it reports only usage, the usage facts must match exactly: the response's, then
//!   Anthropic's or OpenAI's admin usage report for the pool key, the model and the minutes
//!   (filtered by speed and data residency where the dimension is one). The cost must be the rate
//!   table's price of those facts.
//!
//! | Case                          | Dimension                         | Budget  |
//! | ----------------------------- | --------------------------------- | ------- |
//! | `anthropic::geo_us`           | `inference_geo: us` (1.1×)        | < $0.01 |
//! | `anthropic::fast`             | fast mode (2×), Opus 4.8          | < $0.01 |
//! | `anthropic::cache_1h`         | 1-hour cache write, then a read   | ~$0.02  |
//! | `anthropic::web_search`       | web search ($10 / 1K)             | ~$0.03  |
//! | `anthropic::web_fetch`        | web fetch ($0, tokens only)       | ~$0.01  |
//! | `openai::priority`            | Fast mode (`service_tier`)        | < $0.01 |
//! | `openai::flex`                | Flex                              | < $0.01 |
//! | `openai::web_search`          | Responses web search ($10 / 1K)   | ~$0.02  |
//! | `openrouter::generation`      | `usage.cost` + generation API     | < $0.01 |
//! | `openrouter::cancelled`       | a cancelled stream, reconciled    | < $0.01 |
//! | `bedrock::us_profile`         | the `us.` profile's Regional SKUs | < $0.01 |
//! | `xai::web_search`             | xAI tokens + tool fee vs ticks    | ~$0.03  |
//! | `openai::long_context`        | > 272K input: 2× in, 1.5× out     | ~$0.06  |
//! | `xai::long_context`           | ≥ 200K prompt: 2× (vs xAI ticks)  | ~$0.42  |
//! | `bedrock::cancel_gap`         | cancelled-stream estimate gap     | < $0.01 |
//! | `groq::cancel_gap`            | the same on Groq (needs a key)    | < $0.01 |
//!
//! Cached input and every catalog row's basic rates are reconciled elsewhere. BIL-5
//! (`reconcile_live.rs`) does it per batch, against the providers' usage reports, and now prices
//! them. CAT-7 (`catalog_live.rs`) does it per candidate, against reported costs. The long-context tiers
//! are exercised on each vendor's cheapest long-context row (OpenAI's against its usage report,
//! xAI's against its own billed cost); the vectors pin both boundaries exactly (at and one past).
//! The cancel-gap cases measure how far a cancelled stream's row (what the gateway saw) falls
//! short of the full completion a host that keeps generating bills, and record it in
//! `target/verify-cancel-gap.jsonl`.
//!
//! **BIL-25, invoice level.** `VERIFY_INVOICE_ROWS` names a JSONL file of `ai.usage` rows for one
//! pool key. Keep one with `VERIFY_ROWS_OUT` on a live run, or export one from the row store. For
//! each whole UTC day the rows cover, the rows' summed `cost_micros` is compared with the vendor's
//! cost report for the key's Anthropic workspace (`/v1/organizations/cost_report`) or OpenAI
//! project (`/v1/organization/costs`). Both reports are daily and group by workspace or project,
//! not by key. So the comparison holds only when the key's workspace or project carries nothing
//! but that key's traffic, which is the operator's precondition. OpenRouter is reconciled per
//! generation instead (above). Bedrock, Together, Fireworks, Groq and DeepSeek expose no per-key
//! cost API this suite can read, so their cost is held by the rate table and their usage by the
//! response.
//!
//! Every case is listed only with `VERIFY_LIVE=1`, the gateway built, and the keys it needs. A
//! case whose admin key is missing still runs its response-level checks.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[allow(dead_code, unused_imports)]
#[path = "reconcile_live.rs"]
mod recon;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use libtest_mimic::{Arguments, Failed, Trial};
use providers::pricing::{ToolCounts, UsageRow, parse_usd_atto, price};
use recon::{Gateway, Provider, Totals, Window, common};
use serde_json::{Value, json};

/// How long a vendor's usage report or generation lookup may take to show a request. Anthropic
/// documents "typically within 5 minutes"; on 2026-10-10 its report lagged 45-60 minutes (and its
/// admin API's 90-request window can add a `retry-after` of ~20). A report that never shows the
/// request within this is a failure.
const SETTLE: Duration = Duration::from_secs(90 * 60);
/// Anthropic asks for at most one usage-report poll a minute in sustained use, and its admin API
/// allows 90 requests a window shared by the whole organization: poll every three minutes, after
/// a first wait of six for the report to catch up (it lags by up to ~5 minutes).
const POLL: Duration = Duration::from_secs(180);
const FIRST_POLL: Duration = Duration::from_secs(360);

/// Every admin report read in this process goes through one gate, at least this far apart, so
/// cases polling side by side stay inside the providers' admin rate limits.
const ADMIN_SPACING: Duration = Duration::from_secs(15);

fn admin_gate() {
    use std::sync::Mutex;
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(at) = *last {
        let since = at.elapsed();
        if since < ADMIN_SPACING {
            std::thread::sleep(ADMIN_SPACING - since);
        }
    }
    *last = Some(Instant::now());
}

struct Case {
    name: &'static str,
    /// The gateway provider whose pool key the gateway holds.
    provider: &'static str,
    pool_var: &'static str,
    admin_var: Option<&'static str>,
    run: fn(&Ctx) -> Result<(), Failed>,
}

const CASES: &[Case] = &[
    Case {
        name: "anthropic::geo_us",
        provider: "anthropic",
        pool_var: "ANTHROPIC_API_KEY",
        admin_var: Some("ANTHROPIC_ADMIN_KEY"),
        run: anthropic_geo_us,
    },
    Case {
        name: "anthropic::fast",
        provider: "anthropic",
        pool_var: "ANTHROPIC_API_KEY",
        admin_var: Some("ANTHROPIC_ADMIN_KEY"),
        run: anthropic_fast,
    },
    Case {
        name: "anthropic::cache_1h",
        provider: "anthropic",
        pool_var: "ANTHROPIC_API_KEY",
        admin_var: Some("ANTHROPIC_ADMIN_KEY"),
        run: anthropic_cache_1h,
    },
    Case {
        name: "anthropic::web_search",
        provider: "anthropic",
        pool_var: "ANTHROPIC_API_KEY",
        admin_var: Some("ANTHROPIC_ADMIN_KEY"),
        run: anthropic_web_search,
    },
    Case {
        name: "anthropic::web_fetch",
        provider: "anthropic",
        pool_var: "ANTHROPIC_API_KEY",
        admin_var: None,
        run: anthropic_web_fetch,
    },
    Case {
        name: "openai::priority",
        provider: "openai",
        pool_var: "OPENAI_API_KEY",
        admin_var: Some("OPENAI_ADMIN_KEY"),
        run: openai_priority,
    },
    Case {
        name: "openai::flex",
        provider: "openai",
        pool_var: "OPENAI_API_KEY",
        admin_var: None,
        run: openai_flex,
    },
    Case {
        name: "openai::web_search",
        provider: "openai",
        pool_var: "OPENAI_API_KEY",
        admin_var: None,
        run: openai_web_search,
    },
    Case {
        name: "openrouter::generation",
        provider: "openrouter",
        pool_var: "OPENROUTER_API_KEY",
        admin_var: None,
        run: openrouter_generation,
    },
    Case {
        name: "openrouter::cancelled",
        provider: "openrouter",
        pool_var: "OPENROUTER_API_KEY",
        admin_var: None,
        run: openrouter_cancelled,
    },
    Case {
        name: "bedrock::us_profile",
        provider: "bedrock",
        pool_var: "AWS_BEARER_TOKEN_BEDROCK",
        admin_var: None,
        run: bedrock_us_profile,
    },
    Case {
        name: "bedrock::cancel_gap",
        provider: "bedrock",
        pool_var: "AWS_BEARER_TOKEN_BEDROCK",
        admin_var: None,
        run: bedrock_cancel_gap,
    },
    Case {
        name: "groq::cancel_gap",
        provider: "groq",
        pool_var: "GROQ_API_KEY",
        admin_var: None,
        run: groq_cancel_gap,
    },
    Case {
        name: "openai::long_context",
        provider: "openai",
        pool_var: "OPENAI_API_KEY",
        admin_var: Some("OPENAI_ADMIN_KEY"),
        run: openai_long_context,
    },
    Case {
        name: "xai::long_context",
        provider: "xai",
        pool_var: "XAI_API_KEY",
        admin_var: None,
        run: xai_long_context,
    },
    Case {
        name: "xai::web_search",
        provider: "xai",
        pool_var: "XAI_API_KEY",
        admin_var: None,
        run: xai_web_search,
    },
];

fn main() {
    common::started();
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") && recon::gateway_bin().exists() {
        let keys = recon::env_keys();
        for case in CASES {
            let Some(pool) = keys.get(case.pool_var).cloned() else {
                continue;
            };
            let admin = case.admin_var.and_then(|v| keys.get(v).cloned());
            // A case that reads a provider's usage report claims BIL-5 too, which runs it in
            // verify:live's isolated phase; the others are live traffic a window must not see.
            let reconciled = admin.is_some();
            let claims = if reconciled { "BIL-24+BIL-5" } else { "BIL-24" };
            trials.push(Trial::test(
                format!("{claims}::raw::{}", case.name),
                move || {
                    let _traffic = (!reconciled).then(common::live_traffic);
                    common::retrying(|| {
                        let ctx = Ctx::boot(case, &pool, admin.clone())?;
                        (case.run)(&ctx)
                    })
                },
            ));
        }
        if let Some(path) = keys.get("VERIFY_INVOICE_ROWS").cloned() {
            for (provider, pool_var, admin_var) in [
                (
                    Provider::Anthropic,
                    "ANTHROPIC_API_KEY",
                    "ANTHROPIC_ADMIN_KEY",
                ),
                (Provider::OpenAi, "OPENAI_API_KEY", "OPENAI_ADMIN_KEY"),
            ] {
                let (Some(pool), Some(admin)) =
                    (keys.get(pool_var).cloned(), keys.get(admin_var).cloned())
                else {
                    continue;
                };
                let path = PathBuf::from(&path);
                let name = if provider == Provider::Anthropic {
                    "anthropic"
                } else {
                    "openai"
                };
                trials.push(Trial::test(format!("BIL-25::invoice::{name}"), move || {
                    invoice(provider, &pool, &admin, &path)
                }));
            }
        }
    }
    libtest_mimic::run(&args, trials).exit();
}

// ---------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------

struct Ctx {
    case: &'static Case,
    gw: Gateway,
    dir: PathBuf,
    pool: String,
    admin: Option<String>,
}

/// One answered request.
struct Reply {
    status: u16,
    body: String,
    json: Value,
    request_id: Option<String>,
}

impl Ctx {
    fn boot(case: &'static Case, pool: &str, admin: Option<String>) -> Result<Ctx, Failed> {
        let dir = recon::repo_root().join(format!(
            "target/verify-pricing-{}-{}",
            case.name.replace("::", "-"),
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let gw = Gateway::boot(&dir, case.provider, pool)?;
        Ok(Ctx {
            case,
            gw,
            dir,
            pool: pool.to_owned(),
            admin,
        })
    }

    /// POST a managed request to the gateway. `stream_for` cuts the client off after that long
    /// (a cancelled stream).
    fn post(
        &self,
        path: &str,
        headers: &[&str],
        body: &Value,
        stream_for: Option<Duration>,
    ) -> Result<Reply, Failed> {
        let hdr_file = self.dir.join(format!("headers-{}", recon::now_secs()));
        let max_time = stream_for.map_or("180".to_owned(), |d| d.as_secs().max(1).to_string());
        let mut cmd = Command::new("curl");
        cmd.args([
            "-sS",
            "-N",
            "--max-time",
            &max_time,
            "-o",
            "-",
            "-w",
            "\n%{http_code}",
        ])
        .arg("-D")
        .arg(&hdr_file)
        .args([
            "-H",
            "content-type: application/json",
            "--data-binary",
            "@-",
        ]);
        let auth = if path.ends_with("/messages") {
            format!("x-api-key: {}", recon::DEV_TOKEN)
        } else {
            format!("authorization: Bearer {}", recon::DEV_TOKEN)
        };
        cmd.args(["-H", &auth]);
        if path.ends_with("/messages") {
            cmd.args(["-H", "anthropic-version: 2023-06-01"]);
        }
        for h in headers {
            cmd.args(["-H", h]);
        }
        cmd.arg(format!("http://127.0.0.1:{}{path}", self.gw.port));
        let (status, body_text) = match recon::run_curl(cmd, &body.to_string()) {
            Ok(r) => r,
            // A deliberate cut: curl exits 28 (timeout) and prints no status.
            Err(_) if stream_for.is_some() => (0, String::new()),
            Err(e) => return Err(e),
        };
        let hdrs = std::fs::read_to_string(&hdr_file).unwrap_or_default();
        let request_id = hdrs
            .lines()
            .filter_map(|l| l.split_once(':'))
            .filter(|(k, _)| k.trim().eq_ignore_ascii_case("x-beyond-request-id"))
            .map(|(_, v)| v.trim().to_owned())
            .next_back();
        let json = serde_json::from_str(&body_text).unwrap_or(Value::Null);
        Ok(Reply {
            status,
            body: body_text,
            json,
            request_id,
        })
    }

    /// The request's one `ai.usage` row.
    fn row(&self, reply: &Reply) -> Result<Value, Failed> {
        let id = reply.request_id.as_deref().ok_or_else(|| {
            format!(
                "no x-beyond-request-id: HTTP {} {}",
                reply.status, reply.body
            )
        })?;
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let rows: Vec<Value> = recon::usage_rows(&self.gw.log)
                .into_iter()
                .filter(|r| r["request_id"] == id)
                .collect();
            match rows.as_slice() {
                [row] => return Ok(row.clone()),
                [] if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(200)),
                _ => return Err(format!("{id}: {} ai.usage rows, want 1", rows.len()).into()),
            }
        }
    }

    fn ok(&self, reply: &Reply) -> Result<(), Failed> {
        if reply.status == 200 {
            Ok(())
        } else {
            Err(format!("{}: HTTP {}: {}", self.case.name, reply.status, reply.body).into())
        }
    }
}

/// A usage row the rate table prices, rebuilt from a logged `ai.usage` row, with its reported
/// cost (if any) left out when `token_math` (so the tables price the tokens themselves).
fn reprice(row: &Value, token_math: bool) -> Result<providers::pricing::Priced, Failed> {
    let s = |k: &str| row[k].as_str();
    let n = |k: &str| row[k].as_u64().unwrap_or(0);
    let tools = ToolCounts::parse(s("server_tools").unwrap_or("")).map_err(|e| e.to_string())?;
    let wire = if s("usage_wire") == Some("anthropic") {
        providers::WireFormat::Anthropic
    } else {
        providers::WireFormat::OpenAi
    };
    price(&UsageRow {
        price_model: s("price_model"),
        provider: s("provider"),
        price_variant: s("price_variant"),
        usage_wire: wire,
        input_tokens: n("input_tokens"),
        output_tokens: n("output_tokens"),
        cache_read_tokens: n("cache_read_tokens"),
        cache_write_tokens: n("cache_write_tokens"),
        cache_write_1h_tokens: n("cache_write_1h_tokens"),
        gateway_cache_write_tokens: n("gateway_cache_write_tokens"),
        server_tools: tools,
        service_tier: s("service_tier"),
        speed: s("speed"),
        inference_geo: s("inference_geo"),
        upstream_cost_usd: if token_math {
            None
        } else {
            s("upstream_cost_usd")
        },
        upstream_tool_cost_usd: s("upstream_tool_cost_usd"),
        served_by: s("served_by"),
        usage_estimated: row["usage_estimated"] == true,
        upstream_may_continue: row["upstream_may_continue"] == true,
        cache_hit: row["cache_hit"] == true,
        unix_secs: n("start_unix_secs"),
    })
    .map_err(|e| format!("reprice: {e}: {row}").into())
}

/// The row is priced, by this build's table, and its logged amounts are the table's.
fn priced(row: &Value) -> Result<(u64, u64), Failed> {
    if row["price_status"] != "priced" {
        return Err(format!(
            "row not priced: {} {} ({row})",
            row["price_status"], row["price_reason"]
        )
        .into());
    }
    if row["rate_version"] != providers::rates::RATE_VERSION {
        return Err(format!("row priced at another table: {row}").into());
    }
    let p = reprice(row, false)?;
    let (cost, charge) = (
        row["cost_micros"].as_u64().unwrap_or(u64::MAX),
        row["price_micros"].as_u64().unwrap_or(u64::MAX),
    );
    if cost != charge {
        return Err(format!("pass-through: price {charge} is not cost {cost} ({row})").into());
    }
    if (cost, charge) != (p.cost.micros, p.price.micros) {
        return Err(format!(
            "logged cost/price {cost}/{charge} are not the table's {}/{} ({row})",
            p.cost.micros, p.price.micros
        )
        .into());
    }
    Ok((cost, charge))
}

fn eq(what: &str, got: &Value, want: &Value, row: &Value) -> Result<(), Failed> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{what}: row {got}, vendor {want} ({row})").into())
    }
}

/// The Anthropic response's usage block against the row's token facts.
fn anthropic_usage_matches(reply: &Reply, row: &Value) -> Result<(), Failed> {
    let u = &reply.json["usage"];
    let z = |v: &Value| json!(v.as_u64().unwrap_or(0));
    eq(
        "input_tokens",
        &row["input_tokens"],
        &z(&u["input_tokens"]),
        row,
    )?;
    eq(
        "output_tokens",
        &row["output_tokens"],
        &z(&u["output_tokens"]),
        row,
    )?;
    eq(
        "cache_read_tokens",
        &row["cache_read_tokens"],
        &z(&u["cache_read_input_tokens"]),
        row,
    )?;
    eq(
        "cache_write_tokens",
        &row["cache_write_tokens"],
        &z(&u["cache_creation_input_tokens"]),
        row,
    )?;
    eq(
        "cache_write_1h_tokens",
        &row["cache_write_1h_tokens"],
        &z(&u["cache_creation"]["ephemeral_1h_input_tokens"]),
        row,
    )
}

fn tool_count(row: &Value, kind: &str) -> u64 {
    ToolCounts::parse(row["server_tools"].as_str().unwrap_or(""))
        .ok()
        .and_then(|t| providers::pricing::Tool::parse(kind).map(|k| t.get(k)))
        .unwrap_or(0)
}

/// Anthropic's admin usage report for the pool key, the model and `(start, end)`, with extra
/// filters (`inference_geos[]=us`, `speeds[]=fast`), polled until it settles on `want`.
fn anthropic_report_settles(
    ctx: &Ctx,
    model: &str,
    (start, end): (u64, u64),
    filters: &str,
    want: &Totals,
    want_searches: u64,
) -> Result<(), Failed> {
    let Some(admin) = ctx.admin.as_deref() else {
        return Ok(());
    };
    let key_id = recon::pool_key_id(Provider::Anthropic, &ctx.pool, admin)?;
    let deadline = Instant::now() + SETTLE;
    let mut last = String::new();
    std::thread::sleep(FIRST_POLL);
    while Instant::now() < deadline {
        let url = format!(
            "https://api.anthropic.com/v1/organizations/usage_report/messages?starting_at={}\
             &ending_at={}&bucket_width=1m&limit=1440&group_by[]=model&group_by[]=api_key_id\
             &api_key_ids[]={key_id}{filters}",
            recon::rfc3339(start),
            recon::rfc3339(end)
        );
        match anthropic_admin(admin, &url, deadline) {
            Ok(v) => {
                let mut t = Totals::default();
                let mut searches = 0;
                for b in v["data"].as_array().into_iter().flatten() {
                    for r in b["results"].as_array().into_iter().flatten() {
                        if !recon::model_matches(r["model"].as_str().unwrap_or(""), model) {
                            continue;
                        }
                        let n = |v: &Value| v.as_u64().unwrap_or(0);
                        t.fresh += n(&r["uncached_input_tokens"]);
                        t.cache_read += n(&r["cache_read_input_tokens"]);
                        t.cache_write += n(&r["cache_creation"]["ephemeral_5m_input_tokens"])
                            + n(&r["cache_creation"]["ephemeral_1h_input_tokens"]);
                        t.cache_write_1h += n(&r["cache_creation"]["ephemeral_1h_input_tokens"]);
                        t.output += n(&r["output_tokens"]);
                        searches += n(&r["server_tool_use"]["web_search_requests"]);
                    }
                }
                last = format!("{t} 1h={} web_search={searches}", t.cache_write_1h);
                if t.agrees(want)
                    && t.cache_write_1h == want.cache_write_1h
                    && searches == want_searches
                {
                    return Ok(());
                }
            }
            // Keep the last report read: an error says nothing about whether it settled.
            Err(e) if last.is_empty() => last = e.message().unwrap_or("").to_owned(),
            Err(_) => {}
        }
        std::thread::sleep(POLL);
    }
    Err(format!(
        "Anthropic's usage report ({filters}) never settled on the rows: want {want} 1h={} \
         web_search={want_searches}, last {last}",
        want.cache_write_1h
    )
    .into())
}

/// GET an Anthropic admin endpoint with the fast-mode beta (which the `speed` filter needs).
fn anthropic_admin(admin: &str, url: &str, deadline: Instant) -> Result<Value, Failed> {
    // A 429 is the admin API's rate limit, not an answer: wait out its `retry-after` and ask
    // again, while the case's budget allows.
    loop {
        admin_gate();
        match anthropic_admin_once(admin, url) {
            Err(Throttled(wait)) if Instant::now() + wait < deadline => {
                eprintln!(
                    "Anthropic admin API rate-limited; waiting {}s",
                    wait.as_secs()
                );
                std::thread::sleep(wait);
            }
            Err(Throttled(wait)) => {
                return Err(format!(
                    "Anthropic admin API rate-limited (retry-after {}s) past the case's budget",
                    wait.as_secs()
                )
                .into());
            }
            Ok(r) => return r,
        }
    }
}

/// The admin API answered 429; wait this long.
struct Throttled(Duration);

fn anthropic_admin_once(admin: &str, url: &str) -> Result<Result<Value, Failed>, Throttled> {
    let hdr = std::env::temp_dir().join(format!(
        "pricing-live-admin-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let config = format!(
        "header = \"x-api-key: {admin}\"\nheader = \"anthropic-version: 2023-06-01\"\n\
         header = \"anthropic-beta: fast-mode-2026-02-01\"\n"
    );
    let mut cmd = Command::new("curl");
    cmd.args([
        "-sS",
        "--max-time",
        "60",
        "-K",
        "-",
        "-w",
        "\n%{http_code}",
        "-g",
    ])
    .arg("-D")
    .arg(&hdr)
    .arg(url);
    let (status, body) = match recon::run_curl(cmd, &config) {
        Ok(r) => r,
        Err(e) => return Ok(Err(e)),
    };
    if status == 429 {
        let headers = std::fs::read_to_string(&hdr).unwrap_or_default();
        let wait = headers
            .lines()
            .filter_map(|l| l.split_once(':'))
            .find(|(k, _)| k.trim().eq_ignore_ascii_case("retry-after"))
            .and_then(|(_, v)| v.trim().parse::<u64>().ok())
            .unwrap_or(60);
        return Err(Throttled(Duration::from_secs(wait + 1)));
    }
    if status != 200 {
        return Ok(Err(format!("GET {url}: HTTP {status}: {body}").into()));
    }
    Ok(serde_json::from_str(&body).map_err(|e| format!("GET {url}: {e}").into()))
}

/// A row's facts as a [`Totals`] on Anthropic's wire.
fn totals_of(rows: &[&Value]) -> Totals {
    let mut t = Totals::default();
    for r in rows {
        let n = |k: &str| r[k].as_u64().unwrap_or(0);
        t.fresh += n("input_tokens");
        t.cache_read += n("cache_read_tokens");
        t.cache_write += n("cache_write_tokens");
        t.cache_write_1h += n("cache_write_1h_tokens");
        t.output += n("output_tokens");
    }
    t
}

fn messages(model: &str, prompt: &str, extra: Value) -> Value {
    let mut b = json!({
        "model": model, "max_tokens": 64,
        "messages": [{"role": "user", "content": prompt}],
    });
    if let (Some(b), Some(e)) = (b.as_object_mut(), extra.as_object()) {
        b.extend(e.clone());
    }
    b
}

// ---------------------------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------------------------

fn anthropic_geo_us(ctx: &Ctx) -> Result<(), Failed> {
    let model = "claude-sonnet-4-6";
    let window = ctx.admin.as_ref().map(|_| Window::open());
    let reply = ctx.post(
        "/v1/messages",
        &[],
        &messages(model, "Say ok.", json!({"inference_geo": "us"})),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    anthropic_usage_matches(&reply, &row)?;
    eq(
        "inference_geo",
        &row["inference_geo"],
        &reply.json["usage"]["inference_geo"],
        &row,
    )?;
    eq("inference_geo", &row["inference_geo"], &json!("us"), &row)?;
    priced(&row)?;
    if !row["price_detail"]
        .as_str()
        .unwrap_or("")
        .contains("mult=11000")
    {
        return Err(format!("US-only inference not at 1.1x: {row}").into());
    }
    if let Some(w) = window {
        anthropic_report_settles(
            ctx,
            model,
            w.close(),
            "&inference_geos[]=us",
            &totals_of(&[&row]),
            0,
        )?;
    }
    Ok(())
}

fn anthropic_fast(ctx: &Ctx) -> Result<(), Failed> {
    let model = "claude-opus-4-8";
    let window = ctx.admin.as_ref().map(|_| Window::open());
    let reply = ctx.post(
        "/v1/messages",
        &["anthropic-beta: fast-mode-2026-02-01"],
        &messages(model, "Say ok.", json!({"speed": "fast"})),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    anthropic_usage_matches(&reply, &row)?;
    eq("speed", &row["speed"], &json!("fast"), &row)?;
    eq("speed", &row["speed"], &reply.json["usage"]["speed"], &row)?;
    priced(&row)?;
    if !row["price_detail"]
        .as_str()
        .unwrap_or("")
        .starts_with("class=fast")
    {
        return Err(format!("fast mode not priced fast: {row}").into());
    }
    if let Some(w) = window {
        anthropic_report_settles(
            ctx,
            model,
            w.close(),
            "&speeds[]=fast",
            &totals_of(&[&row]),
            0,
        )?;
    }
    Ok(())
}

fn anthropic_cache_1h(ctx: &Ctx) -> Result<(), Failed> {
    let model = "claude-sonnet-5-5";
    // A ~3k-token prefix, unique to this run so the first request writes rather than reads.
    let nonce = format!("{}-{}", std::process::id(), recon::now_secs());
    let prefix = format!(
        "Run {nonce}. {}",
        "The reconciliation suite measures what a one-hour cache write costs. ".repeat(220)
    );
    let body = json!({
        "model": model, "max_tokens": 16,
        "system": [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "messages": [{"role": "user", "content": "Say ok."}],
    });
    let window = ctx.admin.as_ref().map(|_| Window::open());
    let write = ctx.post("/v1/messages", &[], &body, None)?;
    ctx.ok(&write)?;
    let read = ctx.post("/v1/messages", &[], &body, None)?;
    ctx.ok(&read)?;
    let (w, r) = (ctx.row(&write)?, ctx.row(&read)?);
    anthropic_usage_matches(&write, &w)?;
    anthropic_usage_matches(&read, &r)?;
    if w["cache_write_1h_tokens"].as_u64().unwrap_or(0) == 0
        || r["cache_read_tokens"].as_u64().unwrap_or(0) == 0
    {
        return Err(format!("no 1-hour write then read: {w} / {r}").into());
    }
    priced(&w)?;
    priced(&r)?;
    if let Some(win) = window {
        anthropic_report_settles(ctx, model, win.close(), "", &totals_of(&[&w, &r]), 0)?;
    }
    Ok(())
}

fn anthropic_web_search(ctx: &Ctx) -> Result<(), Failed> {
    let model = "claude-haiku-4-5";
    let window = ctx.admin.as_ref().map(|_| Window::open());
    let reply = ctx.post(
        "/v1/messages",
        &[],
        &messages(
            model,
            "Use web search once to find today's date in UTC, then answer with the date only.",
            json!({"tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}], "max_tokens": 256}),
        ),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    anthropic_usage_matches(&reply, &row)?;
    let searches = reply.json["usage"]["server_tool_use"]["web_search_requests"]
        .as_u64()
        .unwrap_or(0);
    if searches == 0 || tool_count(&row, "web_search") != searches {
        return Err(format!(
            "web searches: row {}, response {searches} ({row})",
            row["server_tools"]
        )
        .into());
    }
    priced(&row)?;
    let tools = reprice(&row, false)?.price.parts.tools;
    if tools != searches * 10_000 {
        return Err(format!("web search priced {tools} µ$ for {searches}: {row}").into());
    }
    if let Some(w) = window {
        anthropic_report_settles(ctx, model, w.close(), "", &totals_of(&[&row]), searches)?;
    }
    Ok(())
}

fn anthropic_web_fetch(ctx: &Ctx) -> Result<(), Failed> {
    let reply = ctx.post(
        "/v1/messages",
        &[],
        &messages(
            "claude-haiku-4-5",
            "Fetch https://example.com once and answer with its page title only.",
            json!({"tools": [{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}], "max_tokens": 256}),
        ),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    anthropic_usage_matches(&reply, &row)?;
    let fetches = reply.json["usage"]["server_tool_use"]["web_fetch_requests"]
        .as_u64()
        .unwrap_or(0);
    if fetches == 0 || tool_count(&row, "web_fetch") != fetches {
        return Err(format!(
            "web fetches: row {}, response {fetches} ({row})",
            row["server_tools"]
        )
        .into());
    }
    priced(&row)?;
    if reprice(&row, false)?.price.parts.tools != 0 {
        return Err(format!("web fetch carries a fee: {row}").into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------------------------

fn chat(model: &str, prompt: &str, extra: Value) -> Value {
    let mut b = json!({"model": model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 256});
    if let (Some(b), Some(e)) = (b.as_object_mut(), extra.as_object()) {
        b.extend(e.clone());
    }
    b
}

/// The OpenAI Chat response's usage against the row's token facts.
fn openai_usage_matches(reply: &Reply, row: &Value) -> Result<(), Failed> {
    let u = &reply.json["usage"];
    let z = |v: &Value| json!(v.as_u64().unwrap_or(0));
    eq(
        "input_tokens",
        &row["input_tokens"],
        &z(&u["prompt_tokens"]),
        row,
    )?;
    eq(
        "output_tokens",
        &row["output_tokens"],
        &z(&u["completion_tokens"]),
        row,
    )?;
    eq(
        "cache_read_tokens",
        &row["cache_read_tokens"],
        &z(&u["prompt_tokens_details"]["cached_tokens"]),
        row,
    )
}

fn openai_priority(ctx: &Ctx) -> Result<(), Failed> {
    let model = "gpt-5-mini";
    let window = ctx.admin.as_ref().map(|_| Window::open());
    let reply = ctx.post(
        "/v1/chat/completions",
        &[],
        &chat(
            model,
            "Say ok.",
            json!({"service_tier": "priority", "reasoning_effort": "minimal"}),
        ),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    openai_usage_matches(&reply, &row)?;
    eq(
        "service_tier",
        &row["service_tier"],
        &reply.json["service_tier"],
        &row,
    )?;
    priced(&row)?;
    // OpenAI may serve a priority request at default (and says so): the price follows what served.
    let want = if row["service_tier"] == "priority" {
        "class=fast"
    } else {
        "class=standard"
    };
    if !row["price_detail"].as_str().unwrap_or("").starts_with(want) {
        return Err(format!(
            "served {} but priced {}: {row}",
            row["service_tier"], row["price_detail"]
        )
        .into());
    }
    if let (Some(w), Some(admin)) = (window, ctx.admin.as_deref()) {
        let (start, end) = w.close();
        let key_id = recon::pool_key_id(Provider::OpenAi, &ctx.pool, admin)?;
        let n = |k: &str| row[k].as_u64().unwrap_or(0);
        let want = Totals {
            fresh: n("input_tokens") - n("cache_read_tokens") - n("cache_write_tokens"),
            cache_read: n("cache_read_tokens"),
            cache_write: n("cache_write_tokens"),
            output: n("output_tokens"),
            requests: Some(1),
            ..Totals::default()
        };
        let deadline = Instant::now() + SETTLE;
        loop {
            let t = recon::openai_usage(admin, &key_id, model, start, end)?;
            if t.agrees(&want) {
                break;
            }
            if Instant::now() > deadline {
                return Err(
                    format!("OpenAI's usage report never settled: want {want}, got {t}").into(),
                );
            }
            std::thread::sleep(POLL);
        }
    }
    Ok(())
}

fn openai_flex(ctx: &Ctx) -> Result<(), Failed> {
    let reply = ctx.post(
        "/v1/chat/completions",
        &[],
        &chat(
            "gpt-5-nano",
            "Say ok.",
            json!({"service_tier": "flex", "reasoning_effort": "minimal"}),
        ),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    openai_usage_matches(&reply, &row)?;
    eq(
        "service_tier",
        &row["service_tier"],
        &reply.json["service_tier"],
        &row,
    )?;
    priced(&row)?;
    let want = if row["service_tier"] == "flex" {
        "class=flex"
    } else {
        "class=standard"
    };
    if !row["price_detail"].as_str().unwrap_or("").starts_with(want) {
        return Err(format!(
            "served {} but priced {}: {row}",
            row["service_tier"], row["price_detail"]
        )
        .into());
    }
    Ok(())
}

fn openai_web_search(ctx: &Ctx) -> Result<(), Failed> {
    let reply = ctx.post(
        "/v1/responses",
        &[],
        &json!({
            "model": "gpt-5-mini", "store": false, "max_output_tokens": 512,
            "reasoning": {"effort": "low"},
            "tools": [{"type": "web_search"}],
            "input": "Use web search once to find today's date in UTC; answer with the date only.",
        }),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    let searches = reply.json["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|i| i["type"] == "web_search_call")
        .filter(|i| matches!(i["action"]["type"].as_str(), Some("search") | None))
        .count() as u64;
    if searches == 0 || tool_count(&row, "web_search") != searches {
        return Err(format!(
            "web searches: row {}, response {searches} ({row})",
            row["server_tools"]
        )
        .into());
    }
    priced(&row)?;
    let tools = reprice(&row, false)?.price.parts.tools;
    if tools != searches * 10_000 {
        return Err(format!("web search priced {tools} µ$ for {searches}: {row}").into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// OpenRouter
// ---------------------------------------------------------------------------------------------

/// `GET /api/v1/generation?id=`, polled until OpenRouter has it.
fn generation(ctx: &Ctx, id: &str) -> Result<Value, Failed> {
    let deadline = Instant::now() + Duration::from_secs(5 * 60);
    loop {
        let mut cmd = Command::new("curl");
        cmd.args(["-sS", "--max-time", "30", "-K", "-", "-w", "\n%{http_code}"])
            .arg(format!("https://openrouter.ai/api/v1/generation?id={id}"));
        let (status, body) = recon::run_curl(
            cmd,
            &format!("header = \"authorization: Bearer {}\"\n", ctx.pool),
        )?;
        if status == 200 {
            let v: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
            if v["data"]["total_cost"].is_number() {
                return Ok(v["data"].clone());
            }
        }
        if Instant::now() > deadline {
            return Err(format!("generation {id}: HTTP {status}: {body}").into());
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

/// A JSON number's own text, exactly (`serde_json` keeps it with `arbitrary_precision` off only
/// as f64, so the decimal is re-read from the raw body).
fn number_text(raw: &str, key: &str) -> Option<String> {
    let at = raw.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = raw[at..].trim_start().strip_prefix(':')?.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '-' | '+')))
        .unwrap_or(rest.len());
    Some(rest[..end].to_owned())
}

/// `usd × 1.055`, half-up to the micro-dollar: what the row's `cost_micros` must be.
fn with_fee(usd: &str) -> Result<u64, Failed> {
    with_fee_atto(parse_usd_atto(usd).map_err(|e| format!("{usd}: {e}"))?)
}

/// A decimal USD amount in the gateway's unit for vendor costs, 1e-10 USD, rounded half-up as
/// the gateway rounds it (`usage::vendor::usd_to_e10`).
fn e10(usd: &str) -> Result<u128, Failed> {
    let atto = parse_usd_atto(usd).map_err(|e| format!("{usd}: {e}"))?;
    Ok((atto + 50_000_000) / 100_000_000)
}

fn with_fee_atto(atto: u128) -> Result<u64, Failed> {
    let num = atto * 10_550;
    let den: u128 = 1_000_000_000_000 * 10_000;
    Ok(u64::try_from((num + den / 2) / den).unwrap())
}

fn openrouter_generation(ctx: &Ctx) -> Result<(), Failed> {
    let reply = ctx.post(
        "/v1/chat/completions",
        &["x-beyond-only: openrouter"],
        &chat(
            "openai/gpt-oss-20b",
            "Say ok.",
            json!({"max_completion_tokens": 64}),
        ),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    priced(&row)?;
    let logged = row["upstream_cost_usd"]
        .as_str()
        .ok_or_else(|| format!("no upstream_cost_usd: {row}"))?;
    let reported = number_text(&reply.body, "cost").ok_or("no usage.cost in the response")?;
    if e10(logged)? != e10(&reported)? {
        return Err(format!("row's cost {logged} is not usage.cost {reported}").into());
    }
    let id = row["upstream_generation_id"]
        .as_str()
        .ok_or_else(|| format!("no generation id: {row}"))?;
    let g = generation(ctx, id)?;
    let total = g["total_cost"].to_string();
    if e10(logged)? != e10(&total)? {
        return Err(
            format!("row's cost {logged} is not the generation's total_cost {total}").into(),
        );
    }
    let want = with_fee(logged)?;
    if row["cost_micros"].as_u64() != Some(want) {
        return Err(format!(
            "cost_micros {} is not total_cost × 1.055 = {want} ({row})",
            row["cost_micros"]
        )
        .into());
    }
    Ok(())
}

fn openrouter_cancelled(ctx: &Ctx) -> Result<(), Failed> {
    let reply = ctx.post(
        "/v1/chat/completions",
        &["x-beyond-only: openrouter"],
        &chat(
            "openai/gpt-oss-20b",
            "Count from 1 to 800, one number per line, nothing else.",
            json!({"stream": true, "max_completion_tokens": 2400}),
        ),
        Some(Duration::from_secs(2)),
    )?;
    let row = ctx.row(&reply)?;
    if row["outcome"] != "client_cancelled" {
        return Err(format!("not a cancel: {row}").into());
    }
    if row["price_status"] != "estimated" {
        return Err(format!("a cancelled OpenRouter row must be estimated: {row}").into());
    }
    let estimate = row["cost_micros"]
        .as_u64()
        .ok_or_else(|| format!("no cost: {row}"))?;
    let id = row["upstream_generation_id"]
        .as_str()
        .ok_or_else(|| format!("no generation id: {row}"))?;
    let g = generation(ctx, id)?;
    let reconciled = with_fee(&g["total_cost"].to_string())?;
    // The reconciled amount is the bill: the row's estimate is what the gateway could see.
    eprintln!(
        "BIL-24 openrouter::cancelled: row estimated {estimate} µ$ ({} output tokens relayed); \
         the generation API bills {reconciled} µ$ (cancelled={}, {} completion tokens, host {})",
        row["output_tokens"], g["cancelled"], g["native_tokens_completion"], g["provider_name"]
    );
    if !g["cancelled"].is_boolean() {
        return Err(format!("the generation API did not say whether it was cancelled: {g}").into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Bedrock and xAI
// ---------------------------------------------------------------------------------------------

fn bedrock_us_profile(ctx: &Ctx) -> Result<(), Failed> {
    let reply = ctx.post(
        "/v1/messages",
        &["x-beyond-only: bedrock"],
        &messages("claude-haiku-4-5", "Say ok.", json!({})),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    anthropic_usage_matches(&reply, &row)?;
    eq("provider", &row["provider"], &json!("bedrock"), &row)?;
    eq(
        "price_variant",
        &row["price_variant"],
        &json!("regional"),
        &row,
    )?;
    let (cost, _) = priced(&row)?;
    // The Regional SKUs are the Global (= Anthropic list) card × 1.1 on every token category,
    // and pass-through bills exactly that.
    let mut as_list = row.clone();
    as_list["provider"] = json!("anthropic");
    as_list["price_variant"] = Value::Null;
    let list = reprice(&as_list, false)?.cost.micros;
    if cost.abs_diff(list * 11 / 10) > 1 {
        return Err(format!("Bedrock us. cost {cost} is not the list {list} × 1.1 ({row})").into());
    }
    Ok(())
}

fn xai_web_search(ctx: &Ctx) -> Result<(), Failed> {
    let reply = ctx.post(
        "/v1/responses",
        &[],
        &json!({
            "model": "grok-4.3", "store": false, "max_output_tokens": 512,
            "tools": [{"type": "web_search"}],
            "input": "Use web search once to find today's date in UTC; answer with the date only.",
        }),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    priced(&row)?;
    if row["cost_basis"] != "reported" || tool_count(&row, "web_search") == 0 {
        return Err(format!("want xAI's own cost and a web search on the row: {row}").into());
    }
    // xAI's ticks are its bill; the rate table's price of the same tokens and tools must equal it.
    let math = reprice(&row, true)?.cost.micros;
    let billed = row["cost_micros"].as_u64().unwrap_or(0);
    if math.abs_diff(billed) > 1 {
        return Err(format!(
            "xAI billed {billed} µ$, the rate table says {math} µ$ for the same facts ({row})"
        )
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// BIL-25: invoice level
// ---------------------------------------------------------------------------------------------

/// The rows' summed `cost_micros` per whole UTC day, against the vendor's daily cost report for
/// the key's workspace (Anthropic) or project (OpenAI).
fn invoice(provider: Provider, pool: &str, admin: &str, rows_path: &Path) -> Result<(), Failed> {
    let name = if provider == Provider::Anthropic {
        "anthropic"
    } else {
        "openai"
    };
    let text =
        std::fs::read_to_string(rows_path).map_err(|e| format!("{}: {e}", rows_path.display()))?;
    let mut by_day: BTreeMap<u64, (u64, usize)> = BTreeMap::new();
    let mut unpriced = 0usize;
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let row = v.get("fields").unwrap_or(&v);
        if row["provider"] != name {
            continue;
        }
        let Some(cost) = row["cost_micros"].as_u64() else {
            unpriced += 1;
            continue;
        };
        let day = row["start_unix_secs"].as_u64().unwrap_or(0) / 86_400;
        let e = by_day.entry(day).or_default();
        e.0 += cost;
        e.1 += 1;
    }
    if by_day.is_empty() {
        return Err(format!("{}: no priced {name} rows", rows_path.display()).into());
    }
    let today = recon::now_secs() / 86_400;
    let mut report = Vec::new();
    let mut bad = false;
    for (&day, &(ours, n)) in by_day.iter().filter(|(d, _)| **d < today) {
        let theirs = match provider {
            Provider::Anthropic => anthropic_cost_day(pool, admin, day)?,
            Provider::OpenAi => openai_cost_day(pool, admin, day)?,
        };
        // Vendors report to a fraction of a cent; allow 0.1% or one cent, whichever is larger.
        let slack = (theirs / 1000).max(10_000);
        let ok = ours.abs_diff(theirs) <= slack;
        bad |= !ok;
        report.push(format!(
            "  {} rows={n:<6} ours={:>12.6} vendor={:>12.6} diff={:>+10.6} {}",
            recon::rfc3339(day * 86_400).split('T').next().unwrap_or(""),
            ours as f64 / 1e6,
            theirs as f64 / 1e6,
            (ours as f64 - theirs as f64) / 1e6,
            if ok { "ok" } else { "MISMATCH" }
        ));
    }
    eprintln!(
        "BIL-25 {name} (USD; {unpriced} unpriced rows not summed):\n{}",
        report.join("\n")
    );
    if bad || unpriced > 0 {
        return Err(format!(
            "BIL-25 {name}: the rows do not reconcile with the vendor's invoice:\n{}",
            report.join("\n")
        )
        .into());
    }
    Ok(())
}

/// Anthropic's cost report for one day, the pool key's workspace, in micro-dollars. Amounts are
/// cents, as decimal strings.
fn anthropic_cost_day(pool: &str, admin: &str, day: u64) -> Result<u64, Failed> {
    let key_id = recon::pool_key_id(Provider::Anthropic, pool, admin)?;
    let key = recon::admin_get(
        Provider::Anthropic,
        admin,
        &format!("https://api.anthropic.com/v1/organizations/api_keys/{key_id}"),
    )?;
    let workspace = key["workspace_id"].clone();
    let url = format!(
        "https://api.anthropic.com/v1/organizations/cost_report?starting_at={}&ending_at={}\
         &group_by[]=workspace_id&limit=1",
        recon::rfc3339(day * 86_400),
        recon::rfc3339((day + 1) * 86_400)
    );
    let v = recon::admin_get(Provider::Anthropic, admin, &url)?;
    let mut atto: u128 = 0;
    for b in v["data"].as_array().into_iter().flatten() {
        for r in b["results"].as_array().into_iter().flatten() {
            if r["workspace_id"] == workspace {
                let cents = r["amount"].as_str().unwrap_or("0");
                atto += parse_usd_atto(cents).map_err(|e| format!("{cents}: {e}"))? / 100;
            }
        }
    }
    Ok(u64::try_from(atto / 1_000_000_000_000).unwrap_or(u64::MAX))
}

/// OpenAI's costs for one day, the pool key's project, in micro-dollars.
fn openai_cost_day(pool: &str, admin: &str, day: u64) -> Result<u64, Failed> {
    let key_id = recon::pool_key_id(Provider::OpenAi, pool, admin)?;
    let projects = recon::admin_get(
        Provider::OpenAi,
        admin,
        "https://api.openai.com/v1/organization/projects?limit=100",
    )?;
    let mut project = None;
    for p in projects["data"].as_array().into_iter().flatten() {
        let pid = p["id"].as_str().unwrap_or("");
        let keys = recon::admin_get(
            Provider::OpenAi,
            admin,
            &format!("https://api.openai.com/v1/organization/projects/{pid}/api_keys/{key_id}"),
        );
        if keys.is_ok() {
            project = Some(pid.to_owned());
            break;
        }
    }
    let project = project.ok_or("the pool key is in no project")?;
    let url = format!(
        "https://api.openai.com/v1/organization/costs?start_time={}&end_time={}&bucket_width=1d\
         &group_by=project_id&project_ids={project}&limit=1",
        day * 86_400,
        (day + 1) * 86_400
    );
    let v = recon::admin_get(Provider::OpenAi, admin, &url)?;
    let mut atto: u128 = 0;
    for b in v["data"].as_array().into_iter().flatten() {
        for r in b["results"].as_array().into_iter().flatten() {
            if r["project_id"] == project.as_str() {
                let usd = r["amount"]["value"].to_string();
                atto += parse_usd_atto(&usd).map_err(|e| format!("{usd}: {e}"))?;
            }
        }
    }
    Ok(u64::try_from(atto / 1_000_000_000_000).unwrap_or(u64::MAX))
}

// ---------------------------------------------------------------------------------------------
// Long-context tiers
// ---------------------------------------------------------------------------------------------

/// A prompt of about `n` tokens on the GPT and Grok tokenizers (` a` is one token each), with a
/// nonce first so no cache serves it.
fn long_prompt(n: usize) -> String {
    format!(
        "Run {}-{}. Reply with the word ok and nothing else.{}",
        std::process::id(),
        recon::now_secs(),
        " a".repeat(n)
    )
}

/// OpenAI's long-context tier (more than 272K input tokens bills the whole request at 2× input
/// and 1.5× output) on its cheapest long-context row: about $0.06.
fn openai_long_context(ctx: &Ctx) -> Result<(), Failed> {
    let model = "gpt-6-luna";
    let window = ctx.admin.as_ref().map(|_| Window::open());
    let reply = ctx.post(
        "/v1/chat/completions",
        &[],
        &chat(
            model,
            &long_prompt(290_000),
            json!({"max_completion_tokens": 64}),
        ),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    openai_usage_matches(&reply, &row)?;
    let input = row["input_tokens"].as_u64().unwrap_or(0);
    if input <= 272_000 {
        return Err(format!("prompt of {input} tokens does not reach the tier: {row}").into());
    }
    priced(&row)?;
    if !row["price_detail"]
        .as_str()
        .unwrap_or("")
        .contains("long=true")
    {
        return Err(format!("{input} input tokens not priced at the long tier: {row}").into());
    }
    if let (Some(w), Some(admin)) = (window, ctx.admin.as_deref()) {
        let (start, end) = w.close();
        let key_id = recon::pool_key_id(Provider::OpenAi, &ctx.pool, admin)?;
        let n = |k: &str| row[k].as_u64().unwrap_or(0);
        let want = Totals {
            fresh: n("input_tokens") - n("cache_read_tokens") - n("cache_write_tokens"),
            cache_read: n("cache_read_tokens"),
            cache_write: n("cache_write_tokens"),
            output: n("output_tokens"),
            requests: Some(1),
            ..Totals::default()
        };
        let deadline = Instant::now() + SETTLE;
        loop {
            let t = recon::openai_usage(admin, &key_id, model, start, end)?;
            if t.agrees(&want) {
                break;
            }
            if Instant::now() > deadline {
                return Err(
                    format!("OpenAI's usage report never settled: want {want}, got {t}").into(),
                );
            }
            std::thread::sleep(POLL);
        }
    }
    Ok(())
}

/// xAI's long-context tier (a prompt reaching 200K bills every token at 2×), held to xAI's own
/// `cost_in_usd_ticks`, on its cheapest row: about $0.42.
fn xai_long_context(ctx: &Ctx) -> Result<(), Failed> {
    let reply = ctx.post(
        "/v1/responses",
        &[],
        &json!({
            "model": "grok-build-0.1", "store": false, "max_output_tokens": 64,
            "input": long_prompt(205_000),
        }),
        None,
    )?;
    ctx.ok(&reply)?;
    let row = ctx.row(&reply)?;
    let input = row["input_tokens"].as_u64().unwrap_or(0);
    if input < 200_000 {
        return Err(format!("prompt of {input} tokens does not reach the tier: {row}").into());
    }
    priced(&row)?;
    let math = reprice(&row, true)?;
    if !math.cost.long {
        return Err(format!("{input} prompt tokens not at the long tier: {row}").into());
    }
    let billed = row["cost_micros"].as_u64().unwrap_or(0);
    if math.cost.micros.abs_diff(billed) > 1 {
        return Err(format!(
            "xAI billed {billed} µ$ for {input} prompt tokens, the rate table's long tier says {} µ$ ({row})",
            math.cost.micros
        )
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Cancelled-stream estimate gap (Bedrock, Groq)
// ---------------------------------------------------------------------------------------------

/// How far a cancelled stream's row falls short of what a host that keeps generating bills. The
/// same deterministic prompt is run once to completion (the full bill) and once cut after a
/// second (the row the gateway writes). The gap is recorded in
/// `target/verify-cancel-gap.jsonl` and printed; the case passes when the cut row is flagged
/// (`estimated`, `upstream_may_continue`), since its tokens are a lower bound by design.
fn cancel_gap(ctx: &Ctx, path: &str, model: &str, headers: &[&str]) -> Result<(), Failed> {
    let prompt = "Count from 1 to 300, one number per line, nothing else.";
    let body = |stream: bool| {
        if path.ends_with("/messages") {
            messages(
                model,
                prompt,
                json!({"max_tokens": 1500, "temperature": 0, "stream": stream}),
            )
        } else {
            chat(
                model,
                prompt,
                json!({"max_completion_tokens": 1500, "temperature": 0, "stream": stream}),
            )
        }
    };
    let full = ctx.post(path, headers, &body(false), None)?;
    ctx.ok(&full)?;
    let full_row = ctx.row(&full)?;
    let billed = full_row["output_tokens"].as_u64().unwrap_or(0);
    let cut = ctx.post(path, headers, &body(true), Some(Duration::from_secs(1)))?;
    let row = ctx.row(&cut)?;
    let relayed = row["output_tokens"].as_u64().unwrap_or(0);
    let gap = billed.saturating_sub(relayed);
    let line = json!({
        "case": ctx.case.name, "model": model, "full_output_tokens": billed,
        "cut_row_output_tokens": relayed, "gap_tokens": gap,
        "row_share": if billed > 0 { relayed as f64 / billed as f64 } else { 0.0 },
        "cut_cost_micros": row["cost_micros"], "full_cost_micros": full_row["cost_micros"],
        "outcome": row["outcome"], "at": recon::now_secs(),
    });
    eprintln!("BIL-24 {}: {line}", ctx.case.name);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(recon::repo_root().join("target/verify-cancel-gap.jsonl"))
    {
        use std::io::Write as _;
        let _ = writeln!(f, "{line}");
    }
    if row["outcome"] != "client_cancelled" {
        return Err(format!("the cut did not cancel: {row}").into());
    }
    if row["price_status"] != "estimated" || row["upstream_may_continue"] != true {
        return Err(format!(
            "a cancelled stream on a host that keeps generating must be flagged: {row}"
        )
        .into());
    }
    Ok(())
}

fn bedrock_cancel_gap(ctx: &Ctx) -> Result<(), Failed> {
    cancel_gap(
        ctx,
        "/v1/messages",
        "claude-haiku-4-5",
        &["x-beyond-only: bedrock"],
    )
}

fn groq_cancel_gap(ctx: &Ctx) -> Result<(), Failed> {
    cancel_gap(
        ctx,
        "/v1/chat/completions",
        "openai/gpt-oss-120b",
        &["x-beyond-only: groq"],
    )
}
