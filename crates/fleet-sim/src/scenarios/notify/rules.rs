//! The edge's own rules for a callback (E1, E2, E4, E5, E6), graded against stand-in replicas
//! that record every byte the front forwards and answer whatever a case asks.

use super::*;

// ---------------------------------------------------------------------------------------------
// A stand-in replica: records every request it reads, byte for byte, and answers as asked.
// ---------------------------------------------------------------------------------------------

/// How the stand-in answers a request. Chosen per request from its query's `want`, so cases can
/// run concurrently without sharing a script.
#[derive(Clone, Copy)]
enum Want {
    /// This status, with a body naming it.
    Status(u16),
    /// `503` the first time a session id is seen, then `200`.
    UnavailableOnce,
    /// Read the request and never answer; record when the front closes the connection.
    Hang,
}

#[derive(Clone, Default)]
struct Stub {
    addr: Option<Addr>,
    received: Arc<Mutex<Vec<Vec<u8>>>>,
    /// The bytes of every answer it wrote, in order.
    answered: Arc<Mutex<Vec<Vec<u8>>>>,
    /// When a hung request's connection was closed by the front.
    closed: Arc<Mutex<Vec<Instant>>>,
    seen: Arc<Mutex<std::collections::HashSet<String>>>,
}

fn want_of(req: &front::Request) -> Want {
    match front::query_values(&req.query, "want")
        .first()
        .map(String::as_str)
    {
        Some("hang") => Want::Hang,
        Some("503-once") => Want::UnavailableOnce,
        Some(n) => Want::Status(n.parse().unwrap_or(200)),
        None => Want::Status(200),
    }
}

fn stub_answer(status: u16) -> Vec<u8> {
    let body = format!("{{\"stub\":{status}}}");
    format!(
        "HTTP/1.1 {status} Stub\r\nX-Stub: {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

impl Stub {
    async fn start() -> Self {
        let listener = beyond_ai_test_support::ports::tokio_listener().await;
        let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
        let stub = Self {
            addr: Some(Addr::local(port)),
            ..Self::default()
        };
        let me = stub.clone();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let me = me.clone();
                tokio::spawn(async move {
                    let Ok(req) = front::read_request(&mut sock).await else {
                        return;
                    };
                    if let Ok(mut r) = me.received.lock() {
                        r.push(req.raw.clone());
                    }
                    let status = match want_of(&req) {
                        Want::Status(s) => s,
                        Want::UnavailableOnce => {
                            let sid = front::query_values(&req.query, "session_id")
                                .pop()
                                .unwrap_or_default();
                            let first = me.seen.lock().map(|mut s| s.insert(sid)).unwrap_or(false);
                            if first { 503 } else { 200 }
                        }
                        Want::Hang => {
                            let mut b = [0u8; 64];
                            let _ =
                                tokio::time::timeout(Duration::from_secs(90), sock.read(&mut b))
                                    .await;
                            if let Ok(mut c) = me.closed.lock() {
                                c.push(Instant::now());
                            }
                            return;
                        }
                    };
                    let answer = stub_answer(status);
                    if let Ok(mut a) = me.answered.lock() {
                        a.push(answer.clone());
                    }
                    let _ = sock.write_all(&answer).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        stub
    }

    fn target(&self) -> Target {
        Target {
            name: "stub".into(),
            addr: self.addr.clone().unwrap_or_else(|| Addr::local(0)),
        }
    }

    fn received(&self) -> Vec<Vec<u8>> {
        self.received.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

/// A front over a ring holding only `stub`.
async fn stub_front(stub: &Stub) -> Result<Front, String> {
    let ring = Arc::new(Mutex::new(vec![stub.target()]));
    Front::start(ring, None).await
}

// ---------------------------------------------------------------------------------------------
// E1, E2, E4, E5, E6: the edge's own rules.
// ---------------------------------------------------------------------------------------------

/// **E1** — the front classifies by exactly N21's two rules. A callback in any spelling the rule
/// allows is forwarded; a look-alike is refused `400`; a token on anything but a `POST` to
/// `/_beyond/agent` is refused `403`; a query naming `session_id` twice, counting an encoded key as
/// the same key, is refused `400`. Nothing refused reaches a replica, and a user credential a
/// callback carries is neither checked nor used: it is forwarded as it came.
pub async fn job_ended_classified_exactly(_kind: Kind, _history: &Path) -> Outcome {
    let stub = Stub::start().await;
    let front = match stub_front(&stub).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let at = front.authority().to_owned();
    let mut findings = Vec::new();
    let path = agent_path("s1.classify", "");

    let case = async |what: String, raw: Vec<u8>, want: u16, forwarded: bool| {
        let before = stub.received().len();
        let got = send(&at, &raw).await.map(|(r, _)| front::status_of(&r));
        let after = stub.received();
        let reached = after.len() > before;
        let unchanged = !forwarded || after.last().is_some_and(|r| *r == raw);
        finding(
            "E1",
            got == Ok(want) && reached == forwarded && unchanged,
            format!(
                "{what}: answered {got:?} (want {want}), {} a replica{}",
                if reached { "reached" } else { "did not reach" },
                if forwarded && !unchanged {
                    ", but changed"
                } else {
                    ""
                }
            ),
        )
    };

    for auth in [
        format!("Authorization: Bearer {FAKE_TOKEN}"),
        format!("Authorization: bearer {FAKE_TOKEN}"),
        format!("Authorization: BEARER    {FAKE_TOKEN}"),
        format!("Authorization:   Bearer {FAKE_TOKEN} \t"),
    ] {
        findings.push(
            case(
                format!("callback {:?}", auth.replace(FAKE_TOKEN, "<token>")),
                request("POST", &path, std::slice::from_ref(&auth), BODY),
                200,
                true,
            )
            .await,
        );
    }
    for (what, headers) in [
        (
            "a tab after Bearer",
            vec![format!("Authorization: Bearer\t{FAKE_TOKEN}")],
        ),
        (
            "no space after Bearer",
            vec![format!("Authorization: Bearer{FAKE_TOKEN}")],
        ),
        (
            "two Authorization headers",
            vec![bearer(FAKE_TOKEN), bearer(FAKE_TOKEN)],
        ),
        (
            "a user's Authorization beside a look-alike",
            vec![
                "Authorization: Basic dXNlcjpwYXNz".into(),
                format!("Authorization: bearer \t {FAKE_TOKEN}"),
            ],
        ),
    ] {
        findings.push(
            case(
                format!("look-alike, {what}"),
                request("POST", &path, &headers, BODY),
                400,
                false,
            )
            .await,
        );
    }
    for (what, method, target) in [
        ("GET", "GET", path.clone()),
        ("PUT", "PUT", path.clone()),
        (
            "POST to another path",
            "POST",
            "/_beyond/other?session_id=s1.classify&tenant=t1".into(),
        ),
        (
            "POST to the root",
            "POST",
            "/?session_id=s1.classify&tenant=t1".into(),
        ),
    ] {
        findings.push(
            case(
                format!("token on a {what}"),
                request(method, &target, &[bearer(FAKE_TOKEN)], BODY),
                403,
                false,
            )
            .await,
        );
    }
    for (what, query) in [
        (
            "session_id twice",
            "session_id=s1.a&session_id=s1.b&tenant=t1",
        ),
        (
            "session_id twice, one key encoded",
            "session%5Fid=s1.a&session_id=s1.a&tenant=t1",
        ),
        ("no session_id", "tenant=t1"),
    ] {
        findings.push(
            case(
                format!("callback with {what}"),
                request(
                    "POST",
                    &format!("/_beyond/agent?{query}"),
                    &[bearer(FAKE_TOKEN)],
                    BODY,
                ),
                400,
                false,
            )
            .await,
        );
    }
    // A callback that also carries a user's grant and cookie: still a callback, routed without
    // looking at either (the grant is not even well formed), and forwarded with both untouched.
    findings.push(
        case(
            "callback carrying a malformed user grant and a cookie".into(),
            request(
                "POST",
                &path,
                &[
                    bearer(FAKE_TOKEN),
                    "x-beyond-grant: not-a-grant".into(),
                    "Cookie: session=abc".into(),
                ],
                BODY,
            ),
            200,
            true,
        )
        .await,
    );
    report("job-ended-classified-exactly", &findings)
}

/// **E2** — a callback reaches the replica byte for byte as the sandbox sent it: its
/// `Content-Length`, a `Transfer-Encoding` and its chunks, its header order and case, and nothing
/// added — no grant, no tenant, no forwarding headers.
pub async fn job_ended_forwarded_unchanged(_kind: Kind, _history: &Path) -> Outcome {
    let stub = Stub::start().await;
    let front = match stub_front(&stub).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let at = front.authority().to_owned();
    let path = agent_path("s1.forward", "&extra=kept");
    let chunked = format!(
        "POST {path} HTTP/1.1\r\nhost: edge.test\r\nauthorization: Bearer {FAKE_TOKEN}\r\n\
         Transfer-Encoding: chunked\r\ncontent-type: application/json\r\n\r\n\
         10\r\n{{\"type\":\"job_end\r\n{:x}\r\n{}\r\n0\r\n\r\n",
        BODY.len() - 16,
        &BODY[16..]
    )
    .into_bytes();
    let cases = [
        (
            "Content-Length body",
            request(
                "POST",
                &path,
                &[
                    "Content-Type: application/json".into(),
                    bearer(FAKE_TOKEN),
                    "X-Odd-Header:  spaced  ".into(),
                ],
                BODY,
            ),
        ),
        ("chunked body", chunked),
        (
            "Content-Length and Transfer-Encoding both",
            request(
                "POST",
                &path,
                &[
                    bearer(FAKE_TOKEN),
                    format!("Content-Length: {}", BODY.len()),
                    "Transfer-Encoding: identity".into(),
                ],
                BODY,
            ),
        ),
    ];
    let mut findings = Vec::new();
    for (what, raw) in cases {
        let before = stub.received().len();
        let got = send(&at, &raw).await.map(|(r, _)| front::status_of(&r));
        let received = stub.received();
        let forwarded = received.get(before);
        findings.push(finding(
            "E2",
            forwarded.is_some_and(|f| *f == raw) && received.len() == before + 1,
            match forwarded {
                Some(f) if *f == raw => format!(
                    "{what}: the replica read exactly the {} bytes the sandbox sent (answered {got:?})",
                    raw.len()
                ),
                Some(f) => format!(
                    "{what}: the replica read {} bytes, not the {} sent: {:?}",
                    f.len(),
                    raw.len(),
                    String::from_utf8_lossy(f).replace(FAKE_TOKEN, "<token>")
                ),
                None => format!("{what}: nothing reached the replica (answered {got:?})"),
            },
        ));
    }
    report("job-ended-forwarded-unchanged", &findings)
}

/// **E4** — the front answers within 25 s. A replica that never answers gets its request cancelled
/// at 25 s: the sandbox hears `503`, and the replica sees its connection close, so it stops waiting.
pub async fn job_ended_answers_within_25s(_kind: Kind, _history: &Path) -> Outcome {
    let stub = Stub::start().await;
    let front = match stub_front(&stub).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let sent = Instant::now();
    let raw = request(
        "POST",
        &agent_path("s1.hang", "&want=hang"),
        &[bearer(FAKE_TOKEN)],
        BODY,
    );
    let got = send(front.authority(), &raw).await;
    let mut findings = Vec::new();
    match got {
        Ok((answer, took)) => {
            let status = front::status_of(&answer);
            findings.push(finding(
                "E4",
                status == 503 && took <= Duration::from_millis(25_500),
                format!("a silent replica: the sandbox heard {status} after {took:.1?} (want 503 by 25 s)"),
            ));
            findings.push(finding(
                "E4",
                took >= Duration::from_secs(24),
                format!("and not before the budget: {took:.1?}"),
            ));
        }
        Err(e) => findings.push(finding("E4", false, e)),
    }
    // The cancel: the stub sees the front close the upstream connection at the budget.
    let deadline = Instant::now() + Duration::from_secs(3);
    let closed = loop {
        let c = stub
            .closed
            .lock()
            .map(|c| c.first().copied())
            .unwrap_or(None);
        if c.is_some() || Instant::now() >= deadline {
            break c;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    findings.push(finding(
        "E4",
        closed.is_some_and(|c| c.duration_since(sent) <= Duration::from_millis(26_000)),
        match closed {
            Some(c) => format!(
                "the replica's request was cancelled {:.1?} after the callback arrived",
                c.duration_since(sent)
            ),
            None => "the replica's request was never cancelled".into(),
        },
    ));
    report("job-ended-answers-within-25s", &findings)
}

/// **E5** — the replica's status reaches the sandbox unchanged, and so does the rest of its answer:
/// `200`, `400`, `401`, `403`, `404`, `410`, `411` and `413` on the first answer; a `503` that the
/// replica then answers `200` on retry; and, when every answer within the budget is `503`, that
/// `503` itself. Then, through real replicas, a forged token's `401` with the replica's own body.
pub async fn job_ended_status_passes_through(kind: Kind, history: &Path) -> Outcome {
    let stub = Stub::start().await;
    let front = match stub_front(&stub).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let at = front.authority().to_owned();
    let mut findings = Vec::new();
    let statuses = [200u16, 400, 401, 403, 404, 410, 411, 413];
    let mut runs = Vec::new();
    for s in statuses {
        let raw = request(
            "POST",
            &agent_path(&format!("s1.status-{s}"), &format!("&want={s}")),
            &[bearer(FAKE_TOKEN)],
            BODY,
        );
        let at = at.clone();
        runs.push(tokio::spawn(async move { (s, send(&at, &raw).await) }));
    }
    let once = {
        let raw = request(
            "POST",
            &agent_path("s1.status-once", "&want=503-once"),
            &[bearer(FAKE_TOKEN)],
            BODY,
        );
        let at = at.clone();
        tokio::spawn(async move { send(&at, &raw).await })
    };
    let always = {
        let raw = request(
            "POST",
            &agent_path("s1.status-always", "&want=503"),
            &[bearer(FAKE_TOKEN)],
            BODY,
        );
        let at = at.clone();
        tokio::spawn(async move { send(&at, &raw).await })
    };
    for run in runs {
        let Ok((s, got)) = run.await else {
            findings.push(finding("E5", false, "a case panicked"));
            continue;
        };
        let want = stub_answer(s);
        findings.push(match got {
            Ok((answer, _)) => finding(
                "E5",
                answer == want,
                format!(
                    "{s}: the sandbox heard {} {}",
                    front::status_of(&answer),
                    if answer == want {
                        "with the replica's answer unchanged"
                    } else {
                        "with a changed answer"
                    }
                ),
            ),
            Err(e) => finding("E5", false, format!("{s}: {e}")),
        });
    }
    match once.await {
        Ok(Ok((answer, took))) => findings.push(finding(
            "E5",
            answer == stub_answer(200),
            format!(
                "503 then 200 on retry: the sandbox heard {} after {took:.1?}",
                front::status_of(&answer)
            ),
        )),
        other => findings.push(finding("E5", false, format!("503 then 200: {other:?}"))),
    }
    match always.await {
        Ok(Ok((answer, took))) => findings.push(finding(
            "E5",
            answer == stub_answer(503) && took <= Duration::from_millis(25_500),
            format!(
                "503 throughout: the sandbox heard {} after {took:.1?}, {}",
                front::status_of(&answer),
                if answer == stub_answer(503) {
                    "the replica's own 503"
                } else {
                    "not the replica's 503"
                }
            ),
        )),
        other => findings.push(finding("E5", false, format!("503 throughout: {other:?}"))),
    }

    // Through real replicas: a forged token is the replica's to refuse, with its own fixed body.
    let (fleet, _sandboxes, _flags) = match notify_fleet(kind, history, 2, None, &[]).await {
        Ok(x) => x,
        Err(e) if e.starts_with(ATTACHED_SKIP) => {
            findings.push(finding(
                "E5",
                true,
                "real replicas: skipped on an attached fleet",
            ));
            return report("job-ended-status-passes-through", &findings);
        }
        Err(e) => return Outcome::failed_with(e),
    };
    let Some(real) = fleet.front.as_ref() else {
        return Outcome::failed_with("the fleet has no front".into());
    };
    let raw = request(
        "POST",
        &agent_path("s1.forged", ""),
        &[bearer(FAKE_TOKEN), "Content-Type: application/json".into()],
        BODY,
    );
    match send(real.authority(), &raw).await {
        Ok((answer, _)) => {
            let text = String::from_utf8_lossy(&answer);
            findings.push(finding(
                "E5",
                front::status_of(&answer) == 401 && text.ends_with(r#"{"error":"unauthorized"}"#),
                format!(
                    "a forged token through real replicas: the sandbox heard {} with the replica's \
                     body",
                    front::status_of(&answer)
                ),
            ));
        }
        Err(e) => findings.push(finding("E5", false, format!("forged token: {e}"))),
    }
    report("job-ended-status-passes-through", &findings)
}

/// **E6** — at most 20 callbacks a second are forwarded for one session id; the front answers the
/// rest `429` itself. Another session's callbacks in the same second are not held back by it, and
/// the session's own are forwarded again once the second has passed.
pub async fn job_ended_rate_limited_per_session(_kind: Kind, _history: &Path) -> Outcome {
    let stub = Stub::start().await;
    let front = match stub_front(&stub).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let at = front.authority().to_owned();
    let burst = |session: &'static str, n: usize| {
        let at = at.clone();
        async move {
            let raw = request(
                "POST",
                &agent_path(session, ""),
                &[bearer(FAKE_TOKEN)],
                BODY,
            );
            let runs: Vec<_> = (0..n)
                .map(|_| {
                    let (at, raw) = (at.clone(), raw.clone());
                    tokio::spawn(
                        async move { send(&at, &raw).await.map(|(a, _)| front::status_of(&a)) },
                    )
                })
                .collect();
            let mut statuses = Vec::new();
            for r in runs {
                statuses.push(r.await.ok().and_then(Result::ok).unwrap_or(0));
            }
            statuses
        }
    };
    let began = Instant::now();
    let (hot, other) = tokio::join!(burst("s1.hot", 30), burst("s1.other", 5));
    let took = began.elapsed();
    let count = |v: &[u16], s: u16| v.iter().filter(|x| **x == s).count();
    let forwarded_hot = stub
        .received()
        .iter()
        .filter(|r| String::from_utf8_lossy(r).contains("session_id=s1.hot"))
        .count();
    let mut findings = vec![
        finding(
            "E6",
            took < Duration::from_secs(1)
                && count(&hot, 200) == front::PER_SESSION_PER_SECOND
                && count(&hot, 429) == 30 - front::PER_SESSION_PER_SECOND,
            format!(
                "30 callbacks for one session in {took:.0?}: {} forwarded and answered 200, {} \
                 answered 429 by the front",
                count(&hot, 200),
                count(&hot, 429)
            ),
        ),
        finding(
            "E6",
            forwarded_hot == count(&hot, 200),
            format!("the replica read {forwarded_hot} of them"),
        ),
        finding(
            "E6",
            count(&other, 200) == 5,
            format!(
                "another session's 5 in the same second: {} forwarded",
                count(&other, 200)
            ),
        ),
    ];
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let later = burst("s1.hot", 1).await;
    findings.push(finding(
        "E6",
        later == [200],
        format!("a second later the session's next callback is forwarded: {later:?}"),
    ));
    report("job-ended-rate-limited-per-session", &findings)
}
