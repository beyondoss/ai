//! What every live suite shares, so a single `mise run verify:live` is a trustworthy proof:
//!
//! - [`free_port`]: a loopback port no other live process on this host is handed while this
//!   process lives.
//! - [`live_traffic`] and [`isolated`]: a reconciliation window and any other live traffic never
//!   overlap, on this host.
//! - [`sdk_retry`] and [`judge`]: a provider that stays unavailable through the stock SDKs' own
//!   retry budget makes a cell INCONCLUSIVE, never PASS or FAIL.
//!
//! A suite includes it with `#[path = "common/live.rs"] mod common;`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::fs::{File, OpenOptions, TryLockError};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

/// The dev gateway's `[id_signing_keys]` table (kid `1`: 32 bytes of 7), appended after
/// `[signing_keys]`. Without one a managed Responses turn on a GPT row is a 503 (`signed_id.rs`).
pub const DEV_ID_SIGNING_TOML: &str =
    "\n[id_signing_keys]\n1 = \"BwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwcHBwc=\"\n";

use libtest_mimic::Failed;

fn lock_file(dir: &str, name: &str) -> File {
    let dir = std::env::temp_dir().join(dir);
    let _ = std::fs::create_dir_all(&dir);
    let path: PathBuf = dir.join(name);
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

// --- Ports ----------------------------------------------------------------------------------------

/// Ports are handed out from here up to the kernel's ephemeral range. Above it, a port can be any
/// process's outbound connection the moment after a bind check releases it; below this, the
/// well-known service ports (5432, 6379, 8080, 9090) a host may start later.
const PORT_FLOOR: u16 = 20_000;

/// The first port of the kernel's ephemeral range (`ip_local_port_range`; Linux's default 32768).
fn ephemeral_floor() -> u16 {
    std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .filter(|&p| p > PORT_FLOOR)
        .unwrap_or(32_768)
}

/// A loopback port for a gateway, its admin listener or a nats-server, reserved for the life of
/// this process.
///
/// A bind check alone races: two processes can both find a port free, release it, and hand it to
/// two servers. Pingora binds with `SO_REUSEPORT`, so two gateways then share it silently (a
/// request lands on the other one, or a readiness check passes on a stranger's admin listener);
/// a nats-server holding it instead makes Pingora give up after its bind retries. So a port is
/// first reserved with an exclusive `flock` on `$TMPDIR/beyond-verify-ports/<port>`, held until
/// this process exits (the kernel drops it even on SIGKILL), and only then bind-checked, which
/// catches what isn't ours (another service, or a gateway orphaned by a killed test). Every live
/// suite in every worktree on the host takes its ports here, so none can be handed one twice.
pub fn free_port() -> u16 {
    static HELD: Mutex<Vec<File>> = Mutex::new(Vec::new());
    let ceiling = ephemeral_floor();
    let span = u32::from(ceiling - PORT_FLOOR);
    // Start each process somewhere else in the range, so processes rarely probe the same files.
    let start = std::process::id().wrapping_mul(0x9E37_79B1) % span;
    for i in 0..span {
        let port = PORT_FLOOR + ((start + i) % span) as u16;
        let f = lock_file("beyond-verify-ports", &port.to_string());
        // Another process (or this one, on an earlier call: a lock is per open file) holds it.
        if f.try_lock().is_err() {
            continue;
        }
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            HELD.lock().unwrap().push(f);
            return port;
        }
    }
    panic!("no free loopback port in {PORT_FLOOR}..{ceiling}");
}

// --- Isolation ------------------------------------------------------------------------------------

/// Held by a process (or trial) sending live traffic that no reconciliation can tell from its own.
pub struct Traffic(#[allow(dead_code)] File);

/// Held by a reconciled trial from the start of its window to its end.
pub struct Isolation(#[allow(dead_code)] File);

const LOCK_DIR: &str = "beyond-verify-live";

/// Taken before sending live traffic. Waits while a reconciliation window is open anywhere on
/// the host, then holds off any new window until dropped.
///
/// A provider's usage report is per key, model and minute, so a reconciled trial proves anything
/// only if nothing else used its key on that model in its window — and the catalog sweep drives
/// every row, so no model is ever "unused". Two `flock`s keep the two kinds of process apart:
/// traffic holds `traffic` shared, a window holds `window` shared. Traffic proceeds only if,
/// while holding `traffic`, it finds `window` free; a window proceeds only once, holding `window`,
/// it has taken `traffic` exclusively. Whichever checks second sees the other's hold, so the two
/// never run together; traffic never waits on `window` while holding `traffic`, so neither side
/// can deadlock; and a waiting window turns new traffic away, so it is never starved.
pub fn live_traffic() -> Traffic {
    loop {
        let traffic = lock_file(LOCK_DIR, "traffic");
        traffic.lock_shared().expect("flock traffic");
        let window = lock_file(LOCK_DIR, "window");
        match window.try_lock() {
            Ok(()) => return Traffic(traffic),
            Err(TryLockError::WouldBlock) => {
                drop(traffic);
                eprintln!("live traffic: waiting for an open reconciliation window to close");
                // Exclusive, so it returns only once every window (and every other checker) is
                // gone; released at once.
                window.lock().expect("flock window");
            }
            Err(TryLockError::Error(e)) => panic!("flock window: {e}"),
        }
    }
}

/// Taken by a reconciled trial before its window opens; waits for every [`Traffic`] holder on the
/// host to finish. Other windows may be open at once (each reconciles its own model).
pub fn isolated() -> Isolation {
    let window = lock_file(LOCK_DIR, "window");
    window.lock_shared().expect("flock window");
    let traffic = lock_file(LOCK_DIR, "traffic");
    if traffic.try_lock().is_err() {
        eprintln!("reconciliation: waiting for other live traffic on this host to finish");
        traffic.lock().expect("flock traffic");
    }
    // New traffic now sees `window` held and waits; `traffic` itself needn't stay locked.
    Isolation(window)
}

// --- Provider unavailability ------------------------------------------------------------------

/// The stock OpenAI and Anthropic SDKs' retry policy (openai-python and anthropic-sdk-python
/// `_constants.py` and `_base_client.py`): two retries; Retry-After honored up to two minutes,
/// else 0.5s doubling to at most 8s, less up to 25% jitter.
pub const SDK_MAX_RETRIES: u32 = 2;
const SDK_INITIAL_DELAY: f64 = 0.5;
const SDK_MAX_DELAY: f64 = 8.0;
const SDK_MAX_RETRY_AFTER: f64 = 120.0;

/// Whether the SDK would retry an answer with `status`, and after how long, as retry number
/// `retry` (0-based). `header` reads the answer's response headers; `attempt` is how long the
/// attempt that got this answer took. `None`: the SDK gives up, or the wait and one more attempt
/// as long as this one would not finish inside the test's time budget ([`fits`]), so the cell
/// reports its last answer (INCONCLUSIVE, where it is the provider's) instead of being killed.
pub fn sdk_retry(
    status: u16,
    retry: u32,
    attempt: Duration,
    header: impl Fn(&str) -> Option<String>,
) -> Option<Duration> {
    if retry >= SDK_MAX_RETRIES {
        return None;
    }
    let retry_after = header("retry-after-ms")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|ms| ms / 1000.0)
        .or_else(|| header("retry-after").and_then(|v| v.trim().parse::<f64>().ok()));
    if retry_after.is_some_and(|s| s > SDK_MAX_RETRY_AFTER) {
        return None;
    }
    if !sdk_retryable(status, &header) {
        return None;
    }
    match retry_after {
        Some(s) if s > 0.0 => Some(Duration::from_secs_f64(s)).filter(|&w| fits(w, attempt)),
        _ => sdk_backoff(retry, attempt),
    }
}

/// The SDK's wait before retry number `retry` (0-based) when the server named none, as after a
/// connection error; `None` once its retries are spent, or when the wait and one more `attempt`
/// would not fit the test's time budget.
pub fn sdk_backoff(retry: u32, attempt: Duration) -> Option<Duration> {
    if retry >= SDK_MAX_RETRIES {
        return None;
    }
    let backoff = (SDK_INITIAL_DELAY * 2f64.powi(retry as i32)).min(SDK_MAX_DELAY);
    Some(Duration::from_secs_f64(backoff * (1.0 - 0.25 * jitter()))).filter(|&w| fits(w, attempt))
}

// --- Time budget ----------------------------------------------------------------------------------

/// When this test process started. Each live binary's `main` calls it first, so the clock starts
/// when nextest launched the test (nextest runs one test per process).
pub fn started() -> std::time::Instant {
    static AT: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    *AT.get_or_init(std::time::Instant::now)
}

/// What a cell still does after its last answer: read its billing rows (a catalog cell waits up
/// to five seconds for its row to land, `catalog_live.rs` `ledger`), judge, and stop its gateway.
pub const REPORT_RESERVE: Duration = Duration::from_secs(5);

/// How long nextest lets this test run before it terminates it: the running profile's
/// `slow-timeout` period × `terminate-after`. Read from the workspace's `.config/nextest.toml`
/// for `NEXTEST_PROFILE`: the first of that profile's `[[profile.<name>.overrides]]` whose filter
/// is `binary_id(<this binary>)` (`NEXTEST_BINARY_ID`), else the profile's own `slow-timeout`,
/// following `inherits` and then `default` (overrides are not inherited, as in nextest). `None`
/// outside nextest, or under a profile that never terminates a test.
pub fn test_budget() -> Option<Duration> {
    static BUDGET: std::sync::OnceLock<Option<Duration>> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        let profile = std::env::var("NEXTEST_PROFILE").ok()?;
        let root = std::env::var_os("NEXTEST_WORKSPACE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
        let text = std::fs::read_to_string(root.join(".config/nextest.toml")).ok()?;
        let config: toml::Value = toml::from_str(&text).ok()?;
        budget_in(
            &config,
            &profile,
            std::env::var("NEXTEST_BINARY_ID").ok().as_deref(),
        )
    })
}

/// [`test_budget`] for `profile` and `binary` in a parsed nextest config.
pub fn budget_in(config: &toml::Value, profile: &str, binary: Option<&str>) -> Option<Duration> {
    let profiles = config.get("profile")?;
    let overridden = binary.and_then(|b| {
        let filter = format!("binary_id({b})");
        profiles
            .get(profile)?
            .get("overrides")?
            .as_array()?
            .iter()
            .find(|o| o.get("filter").and_then(toml::Value::as_str) == Some(filter.as_str()))?
            .get("slow-timeout")
            .cloned()
    });
    let mut slow = overridden;
    let mut name = Some(profile.to_owned());
    let mut seen = Vec::new();
    while slow.is_none() {
        let p = name.take()?;
        if seen.contains(&p) {
            return None;
        }
        let table = profiles.get(&p);
        slow = table.and_then(|t| t.get("slow-timeout")).cloned();
        name = table
            .and_then(|t| t.get("inherits"))
            .and_then(toml::Value::as_str)
            .map(str::to_owned)
            .or_else(|| (p != "default").then(|| "default".to_owned()));
        seen.push(p);
    }
    let slow = slow?;
    let period = parse_duration(slow.get("period")?.as_str()?)?;
    let after = u32::try_from(slow.get("terminate-after")?.as_integer()?).ok()?;
    Some(period * after)
}

/// A nextest duration (`"60s"`, `"10m"`, `"1h"`, `"500ms"`).
fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    let split = s.find(|c: char| !c.is_ascii_digit())?;
    let n: u64 = s[..split].parse().ok()?;
    match &s[split..] {
        "ms" => Some(Duration::from_millis(n)),
        "s" => Some(Duration::from_secs(n)),
        "m" => Some(Duration::from_secs(n * 60)),
        "h" => Some(Duration::from_secs(n * 3600)),
        _ => None,
    }
}

/// The time left before nextest terminates this test, less [`REPORT_RESERVE`]; `None` when there
/// is no budget.
pub fn time_left() -> Option<Duration> {
    let budget = test_budget()?;
    Some(budget.saturating_sub(started().elapsed() + REPORT_RESERVE))
}

/// Whether waiting `wait` and then running an attempt as long as `attempt` still ends inside
/// the test's time budget.
pub fn fits(wait: Duration, attempt: Duration) -> bool {
    time_left().is_none_or(|left| wait + attempt <= left)
}

/// What a cell keeps for itself after its client process (a probe, a harness session) ends: read
/// the ledger and judge. A client gets [`time_left`] less this.
pub const CLIENT_RESERVE: Duration = Duration::from_secs(10);

/// How long past its own deadline a client process may run before [`run_client`] kills it.
const CLIENT_GRACE: Duration = Duration::from_secs(5);

/// Runs a cell's client process inside the test's time budget, and returns its exit status and
/// stdout (stderr is the caller's to direct).
///
/// The client is told its deadline in `VERIFY_DEADLINE_SECS` ([`time_left`] less
/// [`CLIENT_RESERVE`]): a harness (`harness.py`, `harness_long.py`) stops the agent it drives
/// there and reports what it saw, with `detail.timed_out`, so the cell can still say what the
/// session was waiting on ([`note_if_stalled_on_provider`]). One that doesn't stop (an SDK probe
/// blocked on a read) is killed with its whole process group [`CLIENT_GRACE`] later, and the cell
/// fails saying so. Either way the cell reports before nextest's kill, never as a bare TIMEOUT.
/// Without a budget (outside nextest) the client runs as long as it likes.
pub fn run_client(
    cmd: &mut std::process::Command,
) -> Result<(std::process::ExitStatus, Vec<u8>), String> {
    run_client_until(
        cmd,
        time_left().map(|left| left.saturating_sub(CLIENT_RESERVE)),
    )
}

/// [`run_client`] with the client's deadline given (`None`: none).
pub fn run_client_until(
    cmd: &mut std::process::Command,
    deadline: Option<Duration>,
) -> Result<(std::process::ExitStatus, Vec<u8>), String> {
    use std::io::Read as _;
    use std::os::unix::process::CommandExt as _;
    if let Some(d) = deadline {
        cmd.env("VERIFY_DEADLINE_SECS", d.as_secs().max(1).to_string());
    }
    let start = std::time::Instant::now();
    let mut child = cmd
        .process_group(0)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let pgid = child.id();
    let mut stdout = child.stdout.take().expect("piped stdout");
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        let _ = stdout.read_to_end(&mut out);
        out
    });
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break status;
        }
        if deadline.is_some_and(|d| start.elapsed() >= d + CLIENT_GRACE) {
            kill_group(pgid);
            let _ = child.wait();
            return Err(format!(
                "the client was still running {}s after its {}s deadline, at the test's time \
                 budget, and was killed",
                CLIENT_GRACE.as_secs(),
                deadline.unwrap_or_default().as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    // Whatever it left behind (an agent's tool subprocess) would hold stdout open.
    kill_group(pgid);
    let out = reader.join().unwrap_or_default();
    Ok((status, out))
}

fn kill_group(pgid: u32) {
    let _ = std::process::Command::new("kill")
        .args(["-KILL", "--", &format!("-{pgid}")])
        .stderr(std::process::Stdio::null())
        .status();
}

/// Calls [`provider_unavailable`] when a client that was stopped at the time budget had been
/// waiting on its provider: the session's last `ai.usage` row is a request the client gave up
/// on (`client_cancelled`) after the provider had all of it (the gateway billed it an estimate,
/// which it does only once the whole body went upstream: `body_delivered` in proxy.rs), and no
/// response head ever reached the client (the gateway logged the upstream request as ending
/// "before response header is sent"), after at least [`STALL`]. A provider that answers this
/// session's other calls in seconds and sends nothing for minutes is unavailable; the gateway
/// relays a head the moment it has one. Only for a route with one live provider.
pub fn note_if_stalled_on_provider(rows: &[serde_json::Value], log: &str) {
    if let Some(why) = stalled_on_provider(rows, log) {
        provider_unavailable(why);
    }
}

/// What [`note_if_stalled_on_provider`] reads: why the session's provider counts as unavailable,
/// or `None` when it doesn't.
pub fn stalled_on_provider(rows: &[serde_json::Value], log: &str) -> Option<String> {
    let row = rows.last()?;
    let id = row["request_id"].as_str()?;
    let provider = row["provider"].as_str()?;
    let ms = row["latency_ms"].as_u64()?;
    let headless = log
        .lines()
        .any(|l| l.contains(id) && l.contains("before response header is sent"));
    (row["outcome"] == "client_cancelled"
        && row["usage_estimated"] == true
        && headless
        && Duration::from_millis(ms) >= STALL)
        .then(|| {
            format!(
                "{provider} had the session's last request ({id}) whole and sent no response \
                 head in {}s; the client was stopped at the test's time budget, and the route \
                 has no other provider to fail over to",
                ms / 1000
            )
        })
}

/// How long a provider must have sent nothing for [`note_if_stalled_on_provider`] to call it
/// unavailable: a minute, against the 1-15s a harness cell's calls take.
pub const STALL: Duration = Duration::from_secs(60);

/// Whether the SDK retries an answer with `status` at all: `x-should-retry` when the server sends
/// it, else 408, 409, 429 and every 5xx.
pub fn sdk_retryable(status: u16, header: impl Fn(&str) -> Option<String>) -> bool {
    match header("x-should-retry").as_deref() {
        Some("true") => true,
        Some("false") => false,
        _ => matches!(status, 408 | 409 | 429) || status >= 500,
    }
}

/// A number in [0, 1): enough to keep concurrent retries from moving in step.
fn jitter() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    f64::from(nanos % 1_000_000) / 1_000_000.0
}

/// How a failed cell's message starts when its provider was unavailable; `verify status` reads a
/// cell whose failure starts with it as not proven, neither passing nor failing.
pub const INCONCLUSIVE: &str = "INCONCLUSIVE: ";

thread_local! {
    static UNAVAILABLE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Records that a provider stayed unavailable through the SDK retry budget on a request only it
/// could serve: the cell this thread is running can't prove anything about the gateway.
pub fn provider_unavailable(why: String) {
    eprintln!("provider unavailable: {why}");
    UNAVAILABLE.with(|u| {
        u.borrow_mut().get_or_insert(why);
    });
}

/// Runs a cell: a failure in a cell that met [`provider_unavailable`] is INCONCLUSIVE, with every
/// problem it found kept in the message.
pub fn judge(cell: impl FnOnce() -> Result<(), Failed>) -> Result<(), Failed> {
    UNAVAILABLE.with(|u| u.borrow_mut().take());
    let result = cell();
    let why = UNAVAILABLE.with(|u| u.borrow_mut().take());
    match (result, why) {
        (Err(e), Some(why)) => Err(inconclusive(&why, e.message().unwrap_or(""))),
        (r, _) => r,
    }
}

/// Runs a whole cell again, after the SDK's backoff, while it fails on [`provider_unavailable`];
/// INCONCLUSIVE once the SDK's retries are spent. For cells whose clients run with retries off
/// and whose answers carry no Retry-After to read.
pub fn retrying(mut attempt: impl FnMut() -> Result<(), Failed>) -> Result<(), Failed> {
    let mut retry = 0;
    loop {
        UNAVAILABLE.with(|u| u.borrow_mut().take());
        let at = std::time::Instant::now();
        let result = attempt();
        let why = UNAVAILABLE.with(|u| u.borrow_mut().take());
        match (result, why) {
            (Err(e), Some(why)) => match sdk_backoff(retry, at.elapsed()) {
                Some(wait) => {
                    eprintln!("retrying the cell in {wait:?}, as the SDK would");
                    std::thread::sleep(wait);
                    retry += 1;
                }
                None => return Err(inconclusive(&why, e.message().unwrap_or(""))),
            },
            (r, _) => return r,
        }
    }
}

/// Calls [`provider_unavailable`] when the last `ai.usage` row a client's session produced is its
/// provider's own retryable answer, relayed: the session ended on the provider being unavailable.
/// Only for a gateway whose route holds one provider, so nothing could have failed over.
pub fn note_if_ended_unavailable(rows: &[serde_json::Value]) {
    let Some(row) = rows.last() else {
        return;
    };
    let status = row["upstream_status"].as_u64().unwrap_or(0) as u16;
    if row["outcome"] == "upstream_error"
        && let Some(provider) = row["provider"].as_str()
        && sdk_retryable(status, |_| None)
    {
        provider_unavailable(format!(
            "{provider} answered {status} to the session's last request ({}), and the route has \
             no other provider to fail over to",
            row["request_id"].as_str().unwrap_or("?")
        ));
    }
}

/// A cell's INCONCLUSIVE failure.
pub fn inconclusive(why: &str, detail: &str) -> Failed {
    format!("{INCONCLUSIVE}{why}\n{detail}").into()
}
