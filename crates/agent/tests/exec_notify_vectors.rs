//! The exec v1.4 (sandbox job notify) vectors, `tests/fixtures/exec/v1.4-*.json`, and the rule that
//! every requirement of the spec has a check.
//!
//! The spec is the module doc of `exec_endpoint` ("Sandbox job notify (v1.4)"), plus FLEET.md's
//! "Job-end callbacks" for the edge. Each requirement there is tagged (`**N1**`, `**E1**`, …) and
//! names its checks. `v1.4-checks.json` lists the same checks per tag, and this file holds the two
//! to each other.
//!
//! Nothing here exercises an implementation; there is none yet. What it does:
//!
//! - loads every v1.4 fixture into a strict schema (`deny_unknown_fields`), so a malformed or
//!   misspelled fixture fails here rather than silently testing nothing later;
//! - recomputes everything in the fixtures that follows from their own inputs: the token's key, kid,
//!   MAC input, tag and token, each token vector's verdict under a reference verifier written from
//!   the spec, the retry timelines from the schedule, the probe answers' meaning, and the callback
//!   body's exact bytes. A vector that disagrees with the spec fails here, before any
//!   implementation is held to it;
//! - fails when a tag in the spec has no check, when the doc and the registry name different
//!   checks, or when a fixture check names a tag no case covers.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use hmac::{Hmac, Mac};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const PROBE: &str = include_str!("../../sandbox/tests/fixtures/exec/v1.4-probe.json");
const KEEP: &str = include_str!("../../sandbox/tests/fixtures/exec/v1.4-keep.json");
const CALLBACK: &str = include_str!("../../sandbox/tests/fixtures/exec/v1.4-callback.json");
const TOKEN: &str = include_str!("../../sandbox/tests/fixtures/exec/v1.4-token.json");
const RETRY: &str = include_str!("../../sandbox/tests/fixtures/exec/v1.4-retry.json");
const CHECKS: &str = include_str!("../../sandbox/tests/fixtures/exec/v1.4-checks.json");
const SPEC_DOC: &str = include_str!("../../sandbox/src/exec_endpoint.rs");
const FLEET_DOC: &str = include_str!("../FLEET.md");

/// Every v1.4 fixture, by file name, so one added to the directory without a loader fails.
const FIXTURES: [(&str, &str); 6] = [
    ("v1.4-probe.json", PROBE),
    ("v1.4-keep.json", KEEP),
    ("v1.4-callback.json", CALLBACK),
    ("v1.4-token.json", TOKEN),
    ("v1.4-retry.json", RETRY),
    ("v1.4-checks.json", CHECKS),
];

// ---- schemas

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeFile {
    spec: String,
    notes: Vec<String>,
    answers: Vec<ProbeCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeCase {
    name: String,
    answer: Value,
    expect: ProbeExpect,
    covers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeExpect {
    native_jobs: bool,
    /// Notify offered natively.
    notify: bool,
    /// Notify offered for the baseline.
    baseline_notify: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeepFile {
    spec: String,
    notes: Vec<String>,
    template: Value,
    cases: Vec<KeepCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeepCase {
    name: String,
    notify: Value,
    expect: KeepExpect,
    covers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeepExpect {
    accepted: bool,
    status: u16,
    recorded: bool,
    refusal: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CallbackFile {
    spec: String,
    notes: Vec<String>,
    keep: CallbackKeep,
    requests: Vec<CallbackRequest>,
    url_session: UrlSession,
    receiver: Vec<ReceiverCase>,
    edge: Vec<EdgeCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CallbackKeep {
    scope: String,
    id: String,
    url: String,
    token: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CallbackRequest {
    name: String,
    method: String,
    url: String,
    headers: BTreeMap<String, String>,
    forbidden_headers: Vec<String>,
    body: String,
    max_body_bytes: usize,
    covers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
/// The tenant and session id the callback's url names.
struct UrlSession {
    tenant: String,
    session_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiverCase {
    name: String,
    request: ReceiverRequest,
    state: ReceiverState,
    expect: ReceiverExpect,
    covers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiverRequest {
    method: String,
    path: String,
    /// The request target's query, after `?`.
    query: String,
    authorization: Vec<String>,
    content_length: Vec<String>,
    transfer_encoding: Vec<String>,
    body: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiverState {
    service_mode: bool,
    /// A session task in this process holds the query's session for the query's tenant.
    open_here: bool,
    /// That session's open reconcile has succeeded.
    open_reconcile_done: bool,
    job: Tracking,
    /// For a proving job: its `keep` with `notify` succeeded.
    keep_succeeded: bool,
    /// For a proving job: the callback arrives inside its proof window.
    in_window: bool,
    sandbox: Listed,
    run_in_flight: bool,
}

/// How the session tracks the job the callback names.
#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum Tracking {
    Proving,
    /// Its stream or a follow has it.
    Held,
    /// v1.4's state: the stream closed after its proof. Not `JobShared::released`.
    Unheld,
    /// A `bash_job` call in flight has it.
    Taken,
    EndRecorded,
    /// The session neither tracks it nor has recorded its end.
    Untracked,
}

/// What a `list` of the scope would say of the job.
#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum Listed {
    Running,
    Exited,
    Ended,
    Lost,
    Missing,
    /// The sandbox cannot be listed.
    Unreachable,
}

/// What the callback does to the job.
#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum Effect {
    None,
    /// The stream is closed and a live reconcile started.
    Release,
    /// Applied as a held stream's `shell_exit`.
    ShellExit,
    /// Today's follow starts.
    Follow,
}

#[derive(Deserialize, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
struct ReceiverExpect {
    status: u16,
    /// The answer waits for a live reconcile's `list`.
    answer_waits_for_list: bool,
    effect: Effect,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgeCase {
    name: String,
    method: String,
    path: String,
    query: String,
    authorization: Vec<String>,
    expect: EdgeExpect,
    covers: Vec<String>,
}

#[derive(Deserialize, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
struct EdgeExpect {
    class: EdgeClass,
    status: Option<u16>,
}

#[derive(Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum EdgeClass {
    Callback,
    User,
    Refused,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenFile {
    spec: String,
    notes: Vec<String>,
    secrets: BTreeMap<String, String>,
    keys: BTreeMap<String, KeyEntry>,
    minted: Minted,
    verifier_keys: Vec<String>,
    vectors: Vec<TokenVector>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyEntry {
    k_hex: String,
    kid: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Minted {
    secret: String,
    purpose: String,
    tenant: String,
    session_id: String,
    scope: String,
    id: String,
    m_hex: String,
    tag_hex: String,
    token: String,
    covers: Vec<String>,
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Context {
    tenant: String,
    session_id: String,
    scope: String,
    id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenVector {
    name: String,
    token: String,
    context: Context,
    expect: Verdict,
    covers: Vec<String>,
}

#[derive(Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum Verdict {
    Valid,
    Malformed,
    UnknownKid,
    BadMac,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetryFile {
    spec: String,
    notes: Vec<String>,
    statuses: Vec<StatusCase>,
    schedule: Schedule,
    timelines: Vec<Timeline>,
}

/// A status code, or an attempt that got none.
#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(untagged)]
enum Answer {
    Status(u16),
    Failure(Failure),
}

#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum Failure {
    ConnectionError,
    TlsError,
    Timeout,
}

#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum Action {
    Done,
    Stop,
    Retry,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StatusCase {
    answer: Answer,
    action: Action,
    covers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Schedule {
    attempts: u32,
    base_s: f64,
    cap_s: f64,
    attempt_timeout_s: f64,
    waits_max_s: Vec<f64>,
    waits_min_s: Vec<f64>,
    twelfth_attempt_start_s: TwelfthStart,
    max_total_s: f64,
    covers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TwelfthStart {
    min: f64,
    max_without_attempt_time: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Timeline {
    name: String,
    outcomes: Vec<Outcome>,
    u: Vec<f64>,
    expect: TimelineExpect,
    covers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Outcome {
    answer: Answer,
    duration_s: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimelineExpect {
    attempts: Vec<Attempt>,
    result: TimelineResult,
    ends_at_s: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Attempt {
    start_s: f64,
    answer: Answer,
    duration_s: f64,
}

#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum TimelineResult {
    Done,
    Stop,
    Exhausted,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChecksFile {
    spec: String,
    notes: Vec<String>,
    requirements: Vec<Requirement>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Requirement {
    id: String,
    checks: Vec<CheckRef>,
}

#[derive(Deserialize, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
struct CheckRef {
    kind: CheckKind,
    name: String,
    task: Option<String>,
}

#[derive(Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
#[serde(rename_all = "kebab-case")]
enum CheckKind {
    Fixture,
    Conformance,
    Test,
    FleetSim,
}

/// The tracker tasks that write checks not written yet. `exists` names one already in the repo.
const TASKS: [&str; 15] = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "C1", "C2", "C3", "D1", "D2", "D3", "E1", "exists",
];

fn parse<T: serde::de::DeserializeOwned>(name: &str, raw: &str) -> T {
    serde_json::from_str(raw).unwrap_or_else(|e| panic!("{name} is malformed: {e}"))
}

fn probe() -> ProbeFile {
    parse("v1.4-probe.json", PROBE)
}
fn keep() -> KeepFile {
    parse("v1.4-keep.json", KEEP)
}
fn callback() -> CallbackFile {
    parse("v1.4-callback.json", CALLBACK)
}
fn token() -> TokenFile {
    parse("v1.4-token.json", TOKEN)
}
fn retry() -> RetryFile {
    parse("v1.4-retry.json", RETRY)
}
fn checks() -> ChecksFile {
    parse("v1.4-checks.json", CHECKS)
}

// ---- loading

#[test]
fn every_v14_fixture_loads_into_its_schema() {
    let p = probe();
    let k = keep();
    let c = callback();
    let t = token();
    let r = retry();
    let ch = checks();
    for (name, spec, notes, cases) in [
        ("probe", &p.spec, &p.notes, p.answers.len()),
        ("keep", &k.spec, &k.notes, k.cases.len()),
        (
            "callback",
            &c.spec,
            &c.notes,
            c.receiver.len() + c.requests.len() + c.edge.len(),
        ),
        ("token", &t.spec, &t.notes, t.vectors.len()),
        (
            "retry",
            &r.spec,
            &r.notes,
            r.statuses.len() + r.timelines.len(),
        ),
        ("checks", &ch.spec, &ch.notes, ch.requirements.len()),
    ] {
        assert!(spec.contains("v1.4"), "{name}: spec names v1.4");
        assert!(!notes.is_empty(), "{name}: has notes");
        assert!(cases > 0, "{name}: has cases");
    }
    // Every case says which requirements it exercises.
    let covers = p
        .answers
        .iter()
        .map(|x| &x.covers)
        .chain(k.cases.iter().map(|x| &x.covers))
        .chain(c.requests.iter().map(|x| &x.covers))
        .chain(c.receiver.iter().map(|x| &x.covers))
        .chain(c.edge.iter().map(|x| &x.covers))
        .chain(t.vectors.iter().map(|x| &x.covers))
        .chain(r.statuses.iter().map(|x| &x.covers))
        .chain(r.timelines.iter().map(|x| &x.covers));
    for list in covers {
        assert!(!list.is_empty(), "a case covers no requirement");
    }
    for case in &k.cases {
        assert!(!case.notify.is_null(), "{}: names a notify", case.name);
    }
    for (name, raw) in FIXTURES {
        // Every case's name is unique within its file: a failure must say which one.
        let v: Value = serde_json::from_str(raw).unwrap();
        let mut names = Vec::new();
        collect_names(&v, &mut names);
        let unique: BTreeSet<&String> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len(), "{name}: case names repeat");
    }
}

#[test]
fn every_v14_fixture_in_the_directory_is_loaded_here() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sandbox/tests/fixtures/exec");
    let on_disk: BTreeSet<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.starts_with("v1.4-"))
        .collect();
    let loaded: BTreeSet<String> = FIXTURES.iter().map(|(n, _)| n.to_string()).collect();
    assert_eq!(on_disk, loaded);
}

fn collect_names<'a>(v: &'a Value, out: &mut Vec<&'a String>) {
    match v {
        Value::Object(m) => {
            if let Some(Value::String(n)) = m.get("name")
                && m.contains_key("covers")
            {
                out.push(n);
            }
            m.values().for_each(|x| collect_names(x, out));
        }
        Value::Array(a) => a.iter().for_each(|x| collect_names(x, out)),
        _ => {}
    }
}

// ---- consistency with the spec

#[test]
fn probe_answers_mean_what_the_spec_says() {
    for case in probe().answers {
        let jobs = case.answer.get("jobs").and_then(Value::as_str);
        let native = jobs == Some("v1.3");
        let v14 = case.answer.get("notify").and_then(Value::as_str) == Some("v1.4");
        let baseline = case.answer.get("jobs").is_none() && v14;
        assert_eq!(case.expect.native_jobs, native, "{}", case.name);
        assert_eq!(case.expect.notify, native && v14, "{}", case.name);
        assert_eq!(case.expect.baseline_notify, baseline, "{}", case.name);
    }
}

#[test]
fn keep_cases_are_consistent() {
    let k = keep();
    assert_eq!(k.template["job"]["op"], "keep");
    assert!(k.template["job"].get("notify").is_none());
    let (mut accepted, mut refused) = (0, 0);
    for case in k.cases {
        let e = &case.expect;
        assert_eq!(
            e.status,
            if e.accepted { 200 } else { 400 },
            "{}",
            case.name
        );
        assert_eq!(
            e.recorded, e.accepted,
            "{}: a refusal records nothing",
            case.name
        );
        assert_eq!(e.refusal.is_some(), !e.accepted, "{}", case.name);
        if e.accepted {
            accepted += 1
        } else {
            refused += 1
        }
    }
    assert!(accepted > 0 && refused > 0);
}

#[test]
fn the_callback_body_is_exactly_type_scope_and_id() {
    let c = callback();
    let want = format!(
        r#"{{"type":"job_ended","scope":"{}","id":"{}"}}"#,
        c.keep.scope, c.keep.id
    );
    assert_eq!(
        c.keep.scope,
        scope_of(&c.url_session.tenant, &c.url_session.session_id),
        "the keep's scope is the url's session's"
    );
    assert!(
        c.keep.url.ends_with(&format!(
            "/_beyond/agent?session_id={}&tenant={}",
            c.url_session.session_id, c.url_session.tenant
        )),
        "the url names the session and the tenant, and nothing else"
    );
    for r in &c.requests {
        assert_eq!(r.method, "POST", "{}", r.name);
        assert_eq!(r.url, c.keep.url, "{}", r.name);
        assert_eq!(r.body, want, "{}", r.name);
        assert!(r.body.len() <= r.max_body_bytes && r.max_body_bytes == 512);
        assert_eq!(
            r.headers.get("Authorization").map(String::as_str),
            Some(format!("Bearer {}", c.keep.token).as_str())
        );
        assert_eq!(
            r.headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
        assert_eq!(
            r.headers.get("Content-Length").map(String::as_str),
            Some(r.body.len().to_string().as_str()),
            "{}: the sender sends Content-Length",
            r.name
        );
        assert_eq!(r.forbidden_headers, ["Cookie", "Transfer-Encoding"]);
    }
}

// ---- the receiver and the edge, from the spec's own rules

/// A header value as N21 reads it: leading and trailing spaces and tabs removed.
fn ows(v: &str) -> &str {
    v.trim_matches([' ', '\t'])
}

/// N21's first rule: exactly one `Authorization` header, `Bearer` in any case, one or more spaces,
/// then `bjn_v1.`. The token, when the request is a callback.
fn callback_token(auth: &[String]) -> Option<&str> {
    let [one] = auth else { return None };
    let v = ows(one);
    let scheme = v.get(..6)?;
    let rest = &v[6..];
    if !scheme.eq_ignore_ascii_case("bearer") || !rest.starts_with(' ') {
        return None;
    }
    let token = rest.trim_start_matches(' ');
    token.starts_with("bjn_v1.").then_some(token)
}

/// N21's second rule: any `Authorization` value that is `bearer` in any case, any run of spaces and
/// tabs, then `bjn_v1.`.
fn looks_like_callback(auth: &[String]) -> bool {
    auth.iter().any(|v| {
        let v = ows(v);
        v.get(..6).is_some_and(|s| s.eq_ignore_ascii_case("bearer"))
            && v[6..]
                .trim_start_matches([' ', '\t'])
                .starts_with("bjn_v1.")
    })
}

/// A v1.3 job id: lowercase hex, 1 to 32 characters.
fn is_job_id(id: &str) -> bool {
    (1..=32).contains(&id.len()) && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// N23 step 2: the query's `session_id` and `tenant`, each exactly once and non-empty.
fn query_session(query: &str) -> Option<UrlSession> {
    let (mut session, mut tenant) = (Vec::new(), Vec::new());
    for (k, v) in url::form_urlencoded::parse(query.as_bytes()) {
        match k.as_ref() {
            "session_id" => session.push(v.into_owned()),
            "tenant" => tenant.push(v.into_owned()),
            _ => {}
        }
    }
    match (session.as_slice(), tenant.as_slice()) {
        ([s], [t]) if !s.is_empty() && !t.is_empty() => Some(UrlSession {
            tenant: t.clone(),
            session_id: s.clone(),
        }),
        _ => None,
    }
}

/// The receiver of N21 and N23, step by step, then N24, N31, N32 and the release row for a
/// callback that passes them all. Nothing here can open a session, because a callback never does
/// (N24): a replica without it answers `503`, and the edge walks on.
fn reference_receiver(keys: &[([u8; 32], String)], case: &ReceiverCase) -> ReceiverExpect {
    let rq = &case.request;
    let st = &case.state;
    let refuse = |status| ReceiverExpect {
        status,
        answer_waits_for_list: false,
        effect: Effect::None,
    };
    // N21: what is a callback, from the head alone; a look-alike is refused before anything else,
    // and job_ended outside a callback too (N22).
    let Some(token) = callback_token(&rq.authorization) else {
        return refuse(if looks_like_callback(&rq.authorization) {
            400
        } else {
            403
        });
    };
    // Step 1.
    if rq.path != "/_beyond/agent" {
        return refuse(404);
    }
    if rq.method != "POST" {
        return refuse(403);
    }
    // Step 2.
    let Some(session) = query_session(&rq.query) else {
        return refuse(400);
    };
    // Step 3.
    match (
        rq.content_length.as_slice(),
        rq.transfer_encoding.is_empty(),
    ) {
        ([], true) => return refuse(411),
        ([n], true) => assert_eq!(
            n.parse::<usize>().ok(),
            Some(rq.body.len()),
            "{}: a case's Content-Length is its body's",
            case.name
        ),
        _ => return refuse(400),
    }
    if rq.body.len() > 4096 {
        return refuse(413);
    }
    let Ok(Value::Object(body)) = serde_json::from_str::<Value>(&rq.body) else {
        return refuse(400);
    };
    if body.get("type").and_then(Value::as_str) != Some("job_ended") {
        return refuse(403);
    }
    let (Some(scope), Some(id)) = (
        body.get("scope").and_then(Value::as_str),
        body.get("id").and_then(Value::as_str),
    ) else {
        return refuse(400);
    };
    if body.len() != 3 || !is_job_id(id) || scope.len() > 64 {
        return refuse(400);
    }
    // Step 4.
    if !st.service_mode {
        return refuse(403);
    }
    // Step 5.
    let cx = Context {
        tenant: session.tenant.clone(),
        session_id: session.session_id.clone(),
        scope: scope.to_string(),
        id: id.to_string(),
    };
    match reference_verify(keys, token, &cx) {
        Verdict::Valid => {}
        Verdict::Malformed | Verdict::BadMac => return refuse(401),
        Verdict::UnknownKid => return refuse(503),
    }
    // Step 6.
    if scope != scope_of(&session.tenant, &session.session_id) {
        return refuse(401);
    }
    let answer = |waits, effect| ReceiverExpect {
        status: 200,
        answer_waits_for_list: waits,
        effect,
    };
    // N24: nobody here holds the session, so nothing happens here.
    if !st.open_here {
        return refuse(503);
    }
    // N32: answered without contacting the sandbox.
    if !st.open_reconcile_done {
        return answer(false, Effect::None);
    }
    match st.job {
        Tracking::Held | Tracking::EndRecorded => answer(false, Effect::None),
        Tracking::Taken => refuse(503),
        Tracking::Untracked => refuse(410),
        // The release row (N35): it waits for no list; the release starts a live reconcile.
        Tracking::Proving if st.in_window && st.keep_succeeded => answer(false, Effect::Release),
        Tracking::Proving => answer(false, Effect::None),
        // N31 and the `unheld` rows.
        Tracking::Unheld => match st.sandbox {
            Listed::Running => answer(true, Effect::None),
            Listed::Exited => answer(true, Effect::ShellExit),
            Listed::Ended | Listed::Lost | Listed::Missing => answer(true, Effect::Follow),
            Listed::Unreachable => ReceiverExpect {
                status: 503,
                answer_waits_for_list: true,
                effect: Effect::Follow,
            },
        },
    }
}

#[test]
fn receiver_cases_follow_from_the_spec() {
    let c = callback();
    let keys = verifier_keys(&token());
    let mut statuses = BTreeSet::new();
    let mut effects = BTreeSet::new();
    for case in &c.receiver {
        assert_eq!(
            reference_receiver(&keys, case),
            case.expect,
            "{}",
            case.name
        );
        if case.state.run_in_flight {
            assert_eq!(
                case.expect.status, 200,
                "{}: served during a run",
                case.name
            );
        }
        statuses.insert(case.expect.status);
        effects.insert(format!("{:?}", case.expect.effect));
    }
    for want in [200, 400, 401, 403, 404, 410, 411, 413, 503] {
        assert!(statuses.contains(&want), "no receiver case answers {want}");
    }
    assert_eq!(effects.len(), 4, "a case for every effect");
    assert!(
        c.receiver
            .iter()
            .any(|x| !x.state.open_here && x.expect.status == 503),
        "a case where no session here takes the callback"
    );
}

/// E1, with N21's rules: how the edge treats a request by its `Authorization` values, and for a
/// callback its method, path and `session_id`. It checks nothing else about a callback (E2).
fn reference_edge(case: &EdgeCase) -> EdgeExpect {
    let refused = |status| EdgeExpect {
        class: EdgeClass::Refused,
        status: Some(status),
    };
    if !looks_like_callback(&case.authorization) {
        return EdgeExpect {
            class: EdgeClass::User,
            status: None,
        };
    }
    if callback_token(&case.authorization).is_none() {
        return refused(400);
    }
    if case.method != "POST" || case.path != "/_beyond/agent" {
        return refused(403);
    }
    let session_ids = url::form_urlencoded::parse(case.query.as_bytes())
        .filter(|(k, _)| k == "session_id")
        .count();
    if session_ids > 1 {
        return refused(400);
    }
    EdgeExpect {
        class: EdgeClass::Callback,
        status: None,
    }
}

#[test]
fn edge_cases_follow_from_the_spec() {
    let c = callback();
    let mut classes = BTreeSet::new();
    for case in &c.edge {
        assert_eq!(reference_edge(case), case.expect, "{}", case.name);
        classes.insert(case.expect.class);
    }
    assert_eq!(classes.len(), 3, "every class has a case");
    // A callback request the edge would forward is one the replica reads a token from: the two
    // classify with the same rule, so they never disagree on bytes the edge passed on unchanged.
    for r in &c.requests {
        let auth = [r.headers["Authorization"].clone()];
        assert!(looks_like_callback(&auth) && callback_token(&auth).is_some());
    }
}

// ---- the token

type HmacSha256 = Hmac<Sha256>;

fn scope_of(tenant: &str, session: &str) -> String {
    let mut h = Sha256::new();
    h.update(b"beyond-jobs\0");
    h.update(tenant.as_bytes());
    h.update(b"\0");
    h.update(session.as_bytes());
    hex::encode(&h.finalize()[..16])
}

fn derive(secret: &[u8]) -> ([u8; 32], String) {
    let mut k = [0u8; 32];
    hkdf::Hkdf::<Sha256>::new(None, secret)
        .expand(b"bjn_v1 notify", &mut k)
        .unwrap();
    let mut h = Sha256::new();
    h.update(b"bjn_v1 kid");
    h.update(k);
    (k, hex::encode(&h.finalize()[..4]))
}

fn lp(out: &mut Vec<u8>, x: &str) {
    out.extend_from_slice(&u32::try_from(x.len()).unwrap().to_be_bytes());
    out.extend_from_slice(x.as_bytes());
}

fn mac_input(
    kid: &str,
    purpose: &str,
    tenant: &str,
    session: &str,
    scope: &str,
    id: &str,
) -> Vec<u8> {
    let mut m = Vec::new();
    for x in ["bjn_v1", kid, purpose, tenant, session, scope, id] {
        lp(&mut m, x);
    }
    m
}

/// The spec's verifier, written from the spec alone: the reference the vectors are held to.
/// N43: the kid and the tag of a token that is exactly the grammar, else `None` (malformed).
fn parse_token(token: &str) -> Option<(&str, Vec<u8>)> {
    let parts: Vec<&str> = token.split('.').collect();
    let [prefix, kid, mac] = parts.as_slice() else {
        return None;
    };
    let hex_kid = kid.len() == 8 && kid.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    let b64url = mac.len() == 43
        && mac
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if *prefix != "bjn_v1" || !hex_kid || !b64url {
        return None;
    }
    let tag = URL_SAFE_NO_PAD.decode(mac).ok()?;
    (tag.len() == 32 && URL_SAFE_NO_PAD.encode(&tag) == *mac).then_some((kid, tag))
}

/// The verifier's keys, as `v1.4-token.json`'s `verifier_keys` names them.
fn verifier_keys(t: &TokenFile) -> Vec<([u8; 32], String)> {
    t.verifier_keys
        .iter()
        .map(|n| derive(&STANDARD.decode(&t.secrets[n]).unwrap()))
        .collect()
}

fn reference_verify(keys: &[([u8; 32], String)], token: &str, cx: &Context) -> Verdict {
    let Some((kid, tag)) = parse_token(token) else {
        return Verdict::Malformed;
    };
    let candidates: Vec<&[u8; 32]> = keys
        .iter()
        .filter(|(_, k)| k == kid)
        .map(|(k, _)| k)
        .collect();
    if candidates.is_empty() {
        return Verdict::UnknownKid;
    }
    let m = mac_input(
        kid,
        "job_ended",
        &cx.tenant,
        &cx.session_id,
        &cx.scope,
        &cx.id,
    );
    for k in candidates {
        let mut h = HmacSha256::new_from_slice(k).unwrap();
        h.update(&m);
        if h.verify_slice(&tag).is_ok() {
            return Verdict::Valid;
        }
    }
    Verdict::BadMac
}

#[test]
fn token_vectors_follow_from_their_inputs() {
    let t = token();
    let mut derived = BTreeMap::new();
    for (name, b64) in &t.secrets {
        let secret = STANDARD.decode(b64).unwrap();
        assert_eq!(secret.len(), 32, "{name}");
        let (k, kid) = derive(&secret);
        let want = &t.keys[name];
        assert_eq!(hex::encode(k), want.k_hex, "{name}: K");
        assert_eq!(kid, want.kid, "{name}: kid");
        derived.insert(name.clone(), (k, kid));
    }
    assert_eq!(t.keys.len(), t.secrets.len());
    let kids: BTreeSet<&String> = derived.values().map(|(_, kid)| kid).collect();
    assert_eq!(kids.len(), derived.len(), "the test keys' kids differ");

    let m = &t.minted;
    assert_eq!(m.purpose, "job_ended");
    assert_eq!(m.scope, scope_of(&m.tenant, &m.session_id));
    let (k, kid) = &derived[&m.secret];
    let input = mac_input(kid, &m.purpose, &m.tenant, &m.session_id, &m.scope, &m.id);
    assert_eq!(hex::encode(&input), m.m_hex, "m");
    let mut h = HmacSha256::new_from_slice(k).unwrap();
    h.update(&input);
    let tag = h.finalize().into_bytes();
    assert_eq!(hex::encode(tag), m.tag_hex, "tag");
    assert_eq!(
        m.token,
        format!("bjn_v1.{kid}.{}", URL_SAFE_NO_PAD.encode(tag))
    );
    assert!(!m.covers.is_empty());

    let keys: Vec<([u8; 32], String)> =
        t.verifier_keys.iter().map(|n| derived[n].clone()).collect();
    let mut seen = BTreeSet::new();
    for v in &t.vectors {
        assert_eq!(
            reference_verify(&keys, &v.token, &v.context),
            v.expect,
            "{}",
            v.name
        );
        seen.insert(v.expect);
    }
    for want in [
        Verdict::Valid,
        Verdict::Malformed,
        Verdict::UnknownKid,
        Verdict::BadMac,
    ] {
        assert!(seen.contains(&want), "a vector with verdict {want:?}");
    }
    assert_eq!(
        t.vectors[0].token, m.token,
        "the first vector is the minted token"
    );
}

// ---- retries

fn action(a: Answer) -> Action {
    match a {
        Answer::Failure(_) => Action::Retry,
        Answer::Status(s) if (200..300).contains(&s) => Action::Done,
        Answer::Status(401 | 403 | 404 | 410) => Action::Stop,
        Answer::Status(407 | 408 | 425 | 429) => Action::Retry,
        Answer::Status(s) if (500..600).contains(&s) => Action::Retry,
        Answer::Status(_) => Action::Stop,
    }
}

fn wait(s: &Schedule, n: u32, u: f64) -> f64 {
    let d = (s.base_s * 2f64.powi(i32::try_from(n).unwrap() - 1)).min(s.cap_s);
    d * (1.0 - u / 2.0)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn retry_vectors_follow_from_the_schedule() {
    let r = retry();
    for st in &r.statuses {
        assert_eq!(action(st.answer), st.action, "{:?}", st.answer);
    }
    let s = &r.schedule;
    assert_eq!(
        (s.attempts, s.base_s, s.cap_s, s.attempt_timeout_s),
        (12, 1.0, 60.0, 30.0)
    );
    let waits: u32 = s.attempts - 1;
    for n in 1..=waits {
        let i = usize::try_from(n - 1).unwrap();
        assert!(close(s.waits_max_s[i], wait(s, n, 0.0)), "wait {n}");
        assert!(close(s.waits_min_s[i], wait(s, n, 1.0)), "wait {n}");
    }
    assert_eq!(s.waits_max_s.len(), usize::try_from(waits).unwrap());
    let max: f64 = s.waits_max_s.iter().sum();
    let min: f64 = s.waits_min_s.iter().sum();
    assert!(close(s.twelfth_attempt_start_s.min, min) && close(min, 181.5));
    assert!(close(s.twelfth_attempt_start_s.max_without_attempt_time, max) && close(max, 363.0));
    assert!(close(
        s.max_total_s,
        max + f64::from(s.attempts) * s.attempt_timeout_s
    ));
    assert!(close(s.max_total_s, 723.0));
    assert!(!s.covers.is_empty());

    for tl in &r.timelines {
        let (mut t, mut result) = (0.0, TimelineResult::Exhausted);
        let mut got = Vec::new();
        for n in 1..=s.attempts {
            let o = &tl.outcomes[usize::try_from(n - 1).unwrap().min(tl.outcomes.len() - 1)];
            let dur = if o.answer == Answer::Failure(Failure::Timeout) {
                s.attempt_timeout_s
            } else {
                o.duration_s
            };
            assert!(
                dur <= s.attempt_timeout_s,
                "{}: an attempt lasts at most 30 s",
                tl.name
            );
            got.push((t, o.answer, dur));
            t += dur;
            match action(o.answer) {
                Action::Done => {
                    result = TimelineResult::Done;
                    break;
                }
                Action::Stop => {
                    result = TimelineResult::Stop;
                    break;
                }
                Action::Retry if n == s.attempts => break,
                Action::Retry => {
                    let u = tl.u[usize::try_from(n - 1).unwrap()];
                    assert!((0.0..1.0).contains(&u), "{}: u in [0, 1)", tl.name);
                    t += wait(s, n, u);
                }
            }
        }
        let e = &tl.expect;
        assert_eq!(e.result, result, "{}", tl.name);
        assert_eq!(e.attempts.len(), got.len(), "{}", tl.name);
        for (a, (start, answer, dur)) in e.attempts.iter().zip(&got) {
            assert!(
                close(a.start_s, *start),
                "{}: {} vs {start}",
                tl.name,
                a.start_s
            );
            assert_eq!(a.answer, *answer, "{}", tl.name);
            assert!(close(a.duration_s, *dur), "{}", tl.name);
        }
        assert!(close(e.ends_at_s, t), "{}", tl.name);
        assert!(t <= s.max_total_s, "{}", tl.name);
    }
}

// ---- every requirement has a check

/// A requirement's text: the tagged bullet `- **N1** …` and its indented continuation lines.
fn requirements(lines: impl Iterator<Item = String>, tag: char) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut open = false;
    let head = format!("- **{tag}");
    for line in lines {
        if let Some(rest) = line.strip_prefix(&head)
            && let Some((num, text)) = rest.split_once("**")
            && !num.is_empty()
            && num.bytes().all(|b| b.is_ascii_digit())
        {
            out.push((format!("{tag}{num}"), text.trim().to_string()));
            open = true;
        } else if open && line.starts_with("  ") && !line.trim().is_empty() {
            let last = out.last_mut().unwrap();
            last.1.push(' ');
            last.1.push_str(line.trim());
        } else {
            open = false;
        }
    }
    out
}

fn spec_requirements() -> Vec<(String, String)> {
    let doc = SPEC_DOC.lines().map_while(|l| {
        l.strip_prefix("//!")
            .map(|r| r.strip_prefix(' ').unwrap_or(r).to_string())
    });
    let mut all = requirements(doc, 'N');
    all.extend(requirements(FLEET_DOC.lines().map(str::to_string), 'E'));
    all
}

/// The `(check: …)` clause that ends a requirement: `kind `name`` items separated by `;`.
fn named_checks(id: &str, text: &str) -> Vec<(CheckKind, String)> {
    let start = text
        .rfind("(check:")
        .unwrap_or_else(|| panic!("{id} names no check"));
    let clause = text[start + "(check:".len()..].trim_end();
    let clause = clause
        .strip_suffix(')')
        .unwrap_or_else(|| panic!("{id}: the check clause must end the requirement"));
    clause
        .split(';')
        .map(|item| {
            let item = item.trim();
            let (kind, rest) = match item.split_once(' ') {
                Some(("conformance", r)) => (CheckKind::Conformance, r),
                Some(("fixture", r)) => (CheckKind::Fixture, r),
                Some(("fleet-sim", r)) => (CheckKind::FleetSim, r),
                _ => (CheckKind::Test, item),
            };
            let name = rest
                .trim()
                .strip_prefix('`')
                .and_then(|r| r.strip_suffix('`'))
                .unwrap_or_else(|| panic!("{id}: check {item:?} is not `name`"));
            (kind, name.to_string())
        })
        .collect()
}

fn covers_in(v: &Value, out: &mut BTreeSet<String>) {
    match v {
        Value::Object(m) => {
            if let Some(Value::Array(c)) = m.get("covers") {
                out.extend(c.iter().map(|x| x.as_str().unwrap().to_string()));
            }
            m.values().for_each(|x| covers_in(x, out));
        }
        Value::Array(a) => a.iter().for_each(|x| covers_in(x, out)),
        _ => {}
    }
}

#[test]
fn every_requirement_has_a_check() {
    let spec = spec_requirements();
    // Tags run N1, N2, … and E1, E2, … with no gap or repeat.
    for tag in ['N', 'E'] {
        let ids: Vec<&String> = spec
            .iter()
            .map(|(id, _)| id)
            .filter(|id| id.starts_with(tag))
            .collect();
        assert!(!ids.is_empty(), "no {tag} requirements found");
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(
                **id,
                format!("{tag}{}", i + 1),
                "requirement tags are sequential"
            );
        }
    }
    let defined: BTreeSet<&String> = spec.iter().map(|(id, _)| id).collect();
    for (id, text) in &spec {
        assert!(
            text.contains("MUST"),
            "{id} is tagged but states no MUST: {text}"
        );
    }

    let mut covered: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for (name, raw) in FIXTURES {
        let mut set = BTreeSet::new();
        covers_in(&serde_json::from_str(raw).unwrap(), &mut set);
        for id in &set {
            assert!(
                defined.contains(id),
                "{name} covers {id}, which the spec does not define"
            );
        }
        covered.insert(name, set);
    }

    let registry = checks();
    let listed: Vec<&String> = registry.requirements.iter().map(|r| &r.id).collect();
    assert_eq!(
        listed,
        spec.iter().map(|(id, _)| id).collect::<Vec<_>>(),
        "v1.4-checks.json lists exactly the spec's requirements, in order"
    );
    for (req, (id, text)) in registry.requirements.iter().zip(&spec) {
        let named = named_checks(id, text);
        let in_registry: Vec<(CheckKind, String)> = req
            .checks
            .iter()
            .map(|c| (c.kind, c.name.clone()))
            .collect();
        assert_eq!(
            in_registry, named,
            "{id}: the doc and v1.4-checks.json name different checks"
        );
        for c in &req.checks {
            match c.kind {
                CheckKind::Fixture => {
                    assert!(c.task.is_none(), "{id}: a fixture is already written");
                    let cases = covered
                        .get(c.name.as_str())
                        .unwrap_or_else(|| panic!("{id}: no fixture named {}", c.name));
                    assert!(cases.contains(id), "{id}: no case in {} covers it", c.name);
                }
                _ => {
                    let task = c
                        .task
                        .as_deref()
                        .unwrap_or_else(|| panic!("{id}: {} has no task", c.name));
                    assert!(TASKS.contains(&task), "{id}: unknown task {task}");
                }
            }
        }
    }
}
