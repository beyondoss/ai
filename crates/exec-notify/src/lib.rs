//! The exec protocol v1.4 **notification kernel**: when a sandbox calls back about a kept job, as
//! one pure function, [`step`].
//!
//! The normative spec is `crates/agent/src/exec_endpoint.rs` ("Sandbox job notify (v1.4)", "When
//! the endpoint calls back", "The answer, and retries"). Everything an endpoint must decide about
//! *when* to call back is here, and nothing else: one notification per job at a time; what is owed,
//! and an answer settling only what was owed when its attempt began; retry or stop by status; the
//! schedule; abandoning what is in flight when a new `keep` arrives; and none of it once the job is
//! removed or its latest `keep` carries no `notify`. Making the request (one `POST`, no redirects, a
//! verified server, 30 s at most) is the caller's, and so is the clock: time arrives as an input,
//! a random draw arrives with the event that needs one, and nothing here does I/O or waits.
//!
//! An endpoint drives it per job: feed every [`Event`] with the time, keep the [`Job`] it returns,
//! and carry out the [`Action`]s: begin an attempt (with the job's notify as recorded *now*),
//! abandon the one in flight, or be woken at a time and feed [`Event::Wake`]. The reference endpoint
//! (`beyond-test-support`'s `exec_mock`) does exactly that; the baseline's shell supervisor
//! follows the same rules in `sh`, and a differential test holds the two together. A model checker
//! calls [`step`] as production does.

/// The most attempts one count makes (N18).
pub const MAX_ATTEMPTS: u32 = 12;

/// How long one attempt may take before it counts as a connection error, in milliseconds (N16).
pub const ATTEMPT_TIMEOUT_MS: u64 = 30_000;

/// One job's notification state. `Default` is a job never kept with `notify`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Job {
    /// The latest `keep` carried `notify`.
    pub notify: bool,
    /// The job was removed (`remove`, `kill` with `remove`, `wipe`): nothing more, ever.
    pub removed: bool,
    /// How many callbacks the job has been owed: one per accepted `keep` with `notify`, one per
    /// state change while such a keep governs.
    pub owed: u64,
    /// How many of those an answer (or the end of a count) has settled.
    pub settled: u64,
    /// The attempt in flight, if any, as the `owed` it carries: what was owed when it began.
    pub in_flight: Option<u64>,
    /// Attempts made in the current count.
    pub attempts: u32,
    /// When the next attempt of the current count may begin, while it waits.
    pub wake_at: Option<u64>,
}

/// What happened to a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    /// An accepted `keep`, with `notify` or without it. With it, the job is owed a callback (the
    /// proof) and whatever was in flight is abandoned; without it, nothing more is sent.
    Keep { notify: bool },
    /// A state change while the job's latest keep carries `notify`: the moment a held stream would
    /// send `shell_exit`, or the job's end becoming observable.
    Change,
    /// The job was removed.
    Removed,
    /// The attempt in flight ended so. `draw` is a fresh uniform `u32`; the wait before a retry is
    /// `d(n) · (1 − u/2)` with `u = draw / 2³²`.
    AttemptEnded { outcome: Outcome, draw: u32 },
    /// A time the kernel asked to be woken at ([`Action::WakeAt`]) has come.
    Wake,
}

/// How one attempt ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// An answer with this status.
    Status(u16),
    /// No status: a connection error, a TLS failure, or no complete answer within 30 s.
    Failed,
}

/// What the endpoint does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Begin attempt `attempt` (1 for a count's first) with the job's notify as recorded now.
    Begin { attempt: u32 },
    /// Abandon the attempt in flight, its connection included.
    Abandon,
    /// Feed [`Event::Wake`] at this time (milliseconds, on the caller's clock).
    WakeAt(u64),
}

/// What an attempt's answer means (N17).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    /// The receiver took the hint: what the attempt carried is settled.
    Done,
    /// It never will, or the request itself is wrong: settled, and not sent again.
    Stop,
    /// Not now: try again, per the schedule.
    Retry,
}

/// The spec's table of answers.
pub fn verdict(outcome: Outcome) -> Verdict {
    match outcome {
        Outcome::Failed => Verdict::Retry,
        Outcome::Status(200..=299) => Verdict::Done,
        Outcome::Status(401 | 403 | 404 | 410) => Verdict::Stop,
        Outcome::Status(407 | 408 | 425 | 429 | 500..=599) => Verdict::Retry,
        Outcome::Status(_) => Verdict::Stop,
    }
}

/// `u` in `[0, 1)` for a draw.
pub fn uniform(draw: u32) -> f64 {
    f64::from(draw) / 4_294_967_296.0
}

/// The wait after failed attempt `n` (1 … 11), in milliseconds: `d(n) · (1 − u/2)` with
/// `d(n) = min(2^(n−1), 60)` s and `u = draw / 2³²` (N18), rounded to the nearest millisecond
/// (half up). Integer arithmetic only, so the baseline's `sh` computes exactly the same number:
/// with `a = d(n)` in ms and `t = a · draw`, it is `a − ⌊t / 2³³⌋`, less one more when the remainder
/// is over `2³²`.
pub fn wait_ms(n: u32, draw: u32) -> u64 {
    let a: u64 = (1u64 << n.saturating_sub(1).min(6)).min(60) * 1000;
    let t = a * u64::from(draw);
    a - (t >> 33) - u64::from((t & ((1 << 33) - 1)) > (1 << 32))
}

/// `keep`'s `notify` grammar (N4): `url` absolute, at most 2048 bytes, a non-empty host, no
/// userinfo, no fragment, and `https` (or `http` to `localhost`, `127.0.0.0/8` or `[::1]`); `token`
/// 1 to 512 bytes of `token68`. The rule broken, if any, never quoting the token. Checked on the
/// bytes as given: a URL parser forgives what the grammar does not (surrounding spaces,
/// `https:///x`).
pub fn check_notify(url: &str, token: &str) -> Result<(), &'static str> {
    check_url(url)?;
    if token.is_empty() || token.len() > 512 {
        return Err("token length");
    }
    let body = token.trim_end_matches('=');
    if body.is_empty()
        || !body
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~+/".contains(&b))
    {
        return Err("token68");
    }
    Ok(())
}

fn check_url(url: &str) -> Result<(), &'static str> {
    if url.len() > 2048 {
        return Err("url length");
    }
    if url.bytes().any(|b| b <= b' ' || b >= 0x7f) {
        return Err("absolute");
    }
    let (scheme, rest) = url.split_once("://").ok_or("absolute")?;
    if !scheme
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphabetic)
        || !scheme
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
    {
        return Err("absolute");
    }
    if url.contains('#') {
        return Err("fragment");
    }
    let authority = &rest[..rest.find(['/', '?']).unwrap_or(rest.len())];
    if authority.contains('@') {
        return Err("userinfo");
    }
    let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
        let end = v6.find(']').ok_or("absolute")?;
        let addr = &v6[..end];
        if addr.parse::<std::net::Ipv6Addr>().is_err() {
            return Err("absolute");
        }
        let after = &v6[end + 1..];
        let port = match after.strip_prefix(':') {
            Some(p) => Some(p),
            None if after.is_empty() => None,
            None => return Err("absolute"),
        };
        (&authority[..end + 2], port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        }
    };
    if host.is_empty() {
        return Err("host");
    }
    if !host.starts_with('[')
        && !host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~%!$&'()*+,;=".contains(&b))
    {
        return Err("absolute");
    }
    if let Some(p) = port
        && !p.is_empty()
        && !(p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u16>().is_ok())
    {
        return Err("absolute");
    }
    match scheme.to_ascii_lowercase().as_str() {
        "https" => Ok(()),
        "http" if loopback(host) => Ok(()),
        _ => Err("scheme"),
    }
}

/// `localhost`, an address in `127.0.0.0/8`, or `[::1]`.
fn loopback(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Some(v6) = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
        return v6.parse::<std::net::Ipv6Addr>() == Ok(std::net::Ipv6Addr::LOCALHOST);
    }
    host.parse::<std::net::Ipv4Addr>()
        .is_ok_and(|a| a.octets()[0] == 127)
}

/// The callback body (N14): exactly `type`, `scope` and `id`, in that order, as JSON strings.
pub fn body(scope: &str, id: &str) -> String {
    let mut out = String::with_capacity(40 + scope.len() + id.len());
    out.push_str("{\"type\":\"job_ended\",\"scope\":");
    json_string(&mut out, scope);
    out.push_str(",\"id\":");
    json_string(&mut out, id);
    out.push('}');
    out
}

fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// One step: the job's next state, and what to do. Pure: the same inputs give the same outputs.
pub fn step(job: Job, event: Event, now: u64) -> (Job, Vec<Action>) {
    let mut j = job;
    let mut out = Vec::new();
    if j.removed {
        return (j, out);
    }
    match event {
        Event::Keep { notify } => {
            if j.in_flight.take().is_some() {
                out.push(Action::Abandon);
            }
            j.wake_at = None;
            j.attempts = 0;
            j.notify = notify;
            if notify {
                j.owed += 1;
                begin(&mut j, &mut out);
            } else {
                j.settled = j.owed;
            }
        }
        Event::Change => {
            if !j.notify {
                return (j, out);
            }
            j.owed += 1;
            if j.in_flight.is_none() {
                // Idle, or waiting out a retry: a new count begins now.
                j.wake_at = None;
                j.attempts = 0;
                begin(&mut j, &mut out);
            }
        }
        Event::Removed => {
            if j.in_flight.take().is_some() {
                out.push(Action::Abandon);
            }
            j = Job {
                removed: true,
                ..Job::default()
            };
        }
        Event::AttemptEnded { outcome, draw } => {
            let Some(carried) = j.in_flight.take() else {
                return (j, out);
            };
            match verdict(outcome) {
                Verdict::Done | Verdict::Stop => {
                    j.settled = j.settled.max(carried);
                    j.attempts = 0;
                }
                Verdict::Retry if j.owed > carried => j.attempts = 0,
                Verdict::Retry if j.attempts >= MAX_ATTEMPTS => {
                    j.settled = j.settled.max(carried);
                    j.attempts = 0;
                }
                Verdict::Retry => {
                    let at = now + wait_ms(j.attempts, draw);
                    j.wake_at = Some(at);
                    out.push(Action::WakeAt(at));
                    return (j, out);
                }
            }
            if j.notify && j.owed > j.settled {
                begin(&mut j, &mut out);
            }
        }
        Event::Wake => {
            if j.in_flight.is_none()
                && j.notify
                && j.owed > j.settled
                && j.wake_at.is_some_and(|at| at <= now)
            {
                begin(&mut j, &mut out);
            }
        }
    }
    (j, out)
}

fn begin(j: &mut Job, out: &mut Vec<Action>) {
    j.wake_at = None;
    j.attempts += 1;
    j.in_flight = Some(j.owed);
    out.push(Action::Begin {
        attempt: j.attempts,
    });
}

/// [`step`] in POSIX `sh` (dash and busybox `ash` alike), for the baseline's job supervisor, which
/// has no Rust to call. Line for line the same rules; `tests.rs` runs the same event sequences
/// through both and compares every state and action.
///
/// State: `KN` notify (0/1), `KR` removed (0/1), `KO` owed, `KS` settled, `KI` the in-flight
/// attempt's carried count (empty: none), `KA` attempts, `KW` wake time (empty: none). Start with
/// `KN=0 KR=0 KO=0 KS=0 KI= KA=0 KW=`. `kstep <event> <now-ms> [<status> <draw>]` with event `k1`
/// (keep with notify), `k0` (keep without), `c` (a change), `r` (removed), `e` (the attempt ended:
/// a 3-digit status, or `f` for no status, and a `u32` draw), or `w` (wake). It leaves the actions
/// in `KX`, space-separated: `b<n>` begin attempt n, `a` abandon, `t<ms>` wake at. Helpers use only
/// `_k`-prefixed names.
pub const SH: &str = r#"kverdict() {
  case $1 in
    f) _kv=r ;;
    2[0-9][0-9]) _kv=d ;;
    401|403|404|410) _kv=s ;;
    407|408|425|429|5[0-9][0-9]) _kv=r ;;
    *) _kv=s ;;
  esac
}
kwait() {
  _ke=$(($1 - 1)); [ "$_ke" -gt 6 ] && _ke=6
  _ka=$((1 << _ke)); [ "$_ka" -gt 60 ] && _ka=60
  _ka=$((_ka * 1000)); _kt=$((_ka * $2))
  _kw=$((_ka - (_kt >> 33)))
  [ $((_kt & 8589934591)) -gt 4294967296 ] && _kw=$((_kw - 1))
  return 0
}
kbegin() { KW=; KA=$((KA + 1)); KI=$KO; KX="${KX}b$KA "; }
kstep() {
  KX=
  [ "$KR" = 1 ] && return 0
  case $1 in
    k1|k0)
      if [ -n "$KI" ]; then KI=; KX="${KX}a "; fi
      KW= KA=0
      if [ "$1" = k1 ]; then KN=1 KO=$((KO + 1)); kbegin; else KN=0 KS=$KO; fi ;;
    c)
      [ "$KN" = 1 ] || return 0
      KO=$((KO + 1))
      if [ -z "$KI" ]; then KW= KA=0; kbegin; fi ;;
    r)
      [ -n "$KI" ] && KX="${KX}a "
      KN=0 KR=1 KO=0 KS=0 KI= KA=0 KW= ;;
    e)
      [ -n "$KI" ] || return 0
      _kc=$KI KI=
      kverdict "$3"
      if [ "$_kv" = r ]; then
        if [ "$KO" -gt "$_kc" ]; then KA=0
        elif [ "$KA" -ge 12 ]; then [ "$_kc" -gt "$KS" ] && KS=$_kc; KA=0
        else kwait "$KA" "$4"; KW=$(($2 + _kw)); KX="${KX}t$KW "; return 0
        fi
      else
        [ "$_kc" -gt "$KS" ] && KS=$_kc; KA=0
      fi
      if [ "$KN" = 1 ] && [ "$KO" -gt "$KS" ]; then kbegin; fi ;;
    w)
      if [ -z "$KI" ] && [ "$KN" = 1 ] && [ "$KO" -gt "$KS" ] && [ -n "$KW" ] && [ "$KW" -le "$2" ]; then kbegin; fi ;;
  esac
  return 0
}
"#;

#[cfg(test)]
mod tests;
