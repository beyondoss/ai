//! The reference v1.4 endpoint (`exec_mock`, native mode) against the spec's endpoint-side
//! requirements and vectors, driven through the agent's own client (`jobs_native`): the probe's
//! answer, `keep`'s `notify` and its grammar (`v1.4-keep.json`), the callback's exact bytes
//! (`v1.4-callback.json`), when it calls back, the retry schedule and the status table over the
//! wire, no redirects, a verified server, and what the mock can be told to get wrong for the proof
//! matrix (lost callbacks, a slow `keep`, no egress, an idle-killing sandbox). When to call back is
//! the shared kernel's (`exec_notify`, re-exported as `exec_endpoint::notify_kernel`), whose own
//! tests run every `v1.4-retry.json` timeline through it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use beyond_ai_agent::exec_endpoint::HttpExecRunner;
use beyond_ai_agent::exec_endpoint::jobs::{
    JobId, JobNotify, JobScope, JobSpec, JobStatus, SandboxJobs,
};
use beyond_ai_agent::exec_endpoint::jobs_native::{JobOffer, NativeJobs};
use common::exec_mock::{Answer, CallbackReceiver, ExecMock, LoseCallbacks, MockOptions};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const KEEP: &str = include_str!("fixtures/exec/v1.4-keep.json");
const CALLBACK: &str = include_str!("fixtures/exec/v1.4-callback.json");

struct Endpoint {
    mock: ExecMock,
    runner: Arc<HttpExecRunner>,
    _dir: tempfile::TempDir,
}

impl Endpoint {
    async fn start(opts: MockOptions) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mock = ExecMock::start_with(dir.path(), opts).await;
        let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
        Self {
            mock,
            runner,
            _dir: dir,
        }
    }

    fn jobs(&self) -> NativeJobs<'_> {
        NativeJobs::over(&self.runner)
    }

    /// Start `script` held in the background and wait until the endpoint lists it. Dropping the
    /// guard drops the hold.
    async fn held(&self, scope: &JobScope, id: &JobId, script: &str) -> Hold {
        let runner = self.runner.clone();
        let (sc, i, spec) = (scope.clone(), id.clone(), spec(script));
        let hold = Hold(tokio::spawn(async move {
            let _ = NativeJobs::over(&runner)
                .run_held(&sc, &i, &spec, &|_| {})
                .await;
        }));
        wait_listed(self, scope, id).await;
        hold
    }
}

struct Hold(tokio::task::JoinHandle<()>);

impl Drop for Hold {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn spec(script: &str) -> JobSpec {
    JobSpec {
        program: "sh".into(),
        args: vec!["-c".into(), script.into()],
        cwd: None,
        label: script.into(),
    }
}

fn scope(name: &str) -> JobScope {
    JobScope::under("/unused", name)
}

fn notify_to(url: &str) -> JobNotify {
    JobNotify {
        url: url.to_string(),
        token: "bjn_v1.4eba5aa3.-dTclPpDTsiFJn-yn7Njt3A4bhaJ7TbBN5JU7DSlyTg".to_string(),
    }
}

/// Wait (bounded) until `scope` lists job `id`.
async fn wait_listed(e: &Endpoint, scope: &JobScope, id: &JobId) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !e
        .jobs()
        .list(scope)
        .await
        .unwrap()
        .iter()
        .any(|j| &j.id == id)
    {
        assert!(Instant::now() < deadline, "the job never started");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A raw `POST` of `body` to the mock; the answer's status.
async fn post(url: &str, body: &Value) -> u16 {
    let addr = url
        .strip_prefix("http://")
        .unwrap()
        .split('/')
        .next()
        .unwrap();
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let body = body.to_string();
    let req = format!(
        "POST /exec HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    sock.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    sock.read_to_end(&mut out).await.unwrap();
    String::from_utf8_lossy(&out)
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}

#[tokio::test]
async fn the_mock_advertises_notify_as_its_deployment_says() {
    let cases = [
        (
            "the setting is off by default",
            MockOptions::default(),
            JobOffer {
                native_jobs: true,
                ..JobOffer::default()
            },
        ),
        (
            "kept jobs survive idling",
            MockOptions::notifying(),
            JobOffer {
                native_jobs: true,
                notify: true,
                baseline_notify: false,
            },
        ),
        (
            "no egress: it knows, and does not advertise",
            MockOptions {
                egress: false,
                ..MockOptions::notifying()
            },
            JobOffer {
                native_jobs: true,
                ..JobOffer::default()
            },
        ),
        (
            "a v1.3 endpoint",
            MockOptions {
                notify: false,
                ..MockOptions::notifying()
            },
            JobOffer {
                native_jobs: true,
                ..JobOffer::default()
            },
        ),
        (
            "an idle-killing sandbox, deployed honestly",
            MockOptions {
                idle_kill: Some(Duration::from_secs(60)),
                ..MockOptions::default()
            },
            JobOffer {
                native_jobs: true,
                ..JobOffer::default()
            },
        ),
        (
            "no native jobs: notify for the baseline",
            MockOptions {
                jobs: false,
                ..MockOptions::notifying()
            },
            JobOffer {
                baseline_notify: true,
                ..JobOffer::default()
            },
        ),
        (
            "no native jobs, the setting off",
            MockOptions {
                jobs: false,
                ..MockOptions::default()
            },
            JobOffer::default(),
        ),
    ];
    for (name, opts, want) in cases {
        let e = Endpoint::start(opts).await;
        assert_eq!(e.runner.job_offer().await, want, "{name}");
        assert_eq!(e.jobs().offers_notify().await, want.notify, "{name}");
    }
}

/// Every case of `v1.4-keep.json`, sent as written to a held job: accepted cases are recorded with
/// their `notify` (the proof is attempted at once), refused ones are a `400` that records nothing.
#[tokio::test]
async fn keep_takes_notify_as_the_vectors_say() {
    // No egress: the vectors' URLs name hosts that must not be dialed, and an attempt without a
    // network still shows which URL the recorded `notify` holds.
    let e = Endpoint::start(MockOptions {
        egress: false,
        ..MockOptions::notifying()
    })
    .await;
    let file: Value = serde_json::from_str(KEEP).unwrap();
    let template = &file["template"];
    let sc = scope(template["job"]["scope"].as_str().unwrap());
    let id = JobId::parse(template["job"]["id"].as_str().unwrap()).unwrap();
    let _hold = e.held(&sc, &id, "sleep 120").await;
    for (i, case) in file["cases"].as_array().unwrap().iter().enumerate() {
        let name = case["name"].as_str().unwrap();
        let before = e.jobs().list(&sc).await.unwrap();
        let attempts_before = e.mock.callbacks().len();
        let mut req = template.clone();
        // A number of its own per case, so a refused keep that recorded anything would show.
        req["job"]["num"] = json!(100 + i);
        req["job"]["notify"] = case["notify"].clone();
        let status = post(&e.mock.url, &req).await;
        let want = &case["expect"];
        assert_eq!(status as u64, want["status"].as_u64().unwrap(), "{name}");
        let after = e.jobs().list(&sc).await.unwrap();
        let job = after.iter().find(|j| j.id == id).unwrap();
        if want["recorded"].as_bool().unwrap() {
            assert!(
                job.kept && job.num == Some(100 + i as u64),
                "{name}: {job:?}"
            );
            let url = case["notify"]["url"].as_str().unwrap();
            let deadline = Instant::now() + Duration::from_secs(2);
            while !e.mock.callbacks()[attempts_before..]
                .iter()
                .any(|a| a.url == url && a.n == 1)
            {
                assert!(
                    Instant::now() < deadline,
                    "{name}: no proof attempt for the recorded notify"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        } else {
            let was = before.iter().find(|j| j.id == id).unwrap();
            assert_eq!(
                (job.kept, job.num, job.kind.clone(), job.seen),
                (was.kept, was.num, was.kind.clone(), was.seen),
                "{name}: a refused keep recorded something"
            );
        }
    }
}

/// The first case of `v1.4-callback.json`: one `POST` to the URL exactly as `keep` gave it, its
/// headers and body byte for byte, nothing forbidden, and the token only in `Authorization`.
#[tokio::test]
async fn the_callback_is_the_vector_byte_for_byte() {
    let file: Value = serde_json::from_str(CALLBACK).unwrap();
    let vector = &file["requests"][0];
    let keep = &file["keep"];
    let rx = CallbackReceiver::start().await;
    let e = Endpoint::start(MockOptions::notifying()).await;
    let sc = scope(keep["scope"].as_str().unwrap());
    let id = JobId::parse(keep["id"].as_str().unwrap()).unwrap();
    let token = keep["token"].as_str().unwrap();
    let _hold = e.held(&sc, &id, "sleep 120").await;
    // The vector's URL with this receiver's origin, and a query of its own to carry exactly.
    let url = format!(
        "http://{}/_beyond/agent?session_id=a07-s3.01J9ZQ4N6V&tenant=acme&x=%2F%20",
        rx.addr
    );
    e.jobs()
        .keep_notify(
            &sc,
            &id,
            3,
            "background",
            120,
            &JobNotify {
                url: url.clone(),
                token: token.to_string(),
            },
        )
        .await
        .unwrap();
    let answered = Instant::now();
    let got = rx.wait_for(1, Duration::from_secs(5)).await;
    assert_eq!(got.len(), 1, "no proof callback");
    let req = &got[0];
    assert!(
        req.at.duration_since(answered) <= Duration::from_secs(2),
        "the proof began more than 2 s after keep was answered"
    );
    assert_eq!(req.method, vector["method"]);
    assert_eq!(
        req.target,
        "/_beyond/agent?session_id=a07-s3.01J9ZQ4N6V&tenant=acme&x=%2F%20"
    );
    for (name, want) in vector["headers"].as_object().unwrap() {
        assert_eq!(
            req.header_values(name),
            vec![want.as_str().unwrap()],
            "{name}"
        );
    }
    for name in vector["forbidden_headers"].as_array().unwrap() {
        assert!(
            req.header_values(name.as_str().unwrap()).is_empty(),
            "{name} sent"
        );
    }
    assert_eq!(req.body, vector["body"].as_str().unwrap().as_bytes());
    assert!(req.body.len() as u64 <= vector["max_body_bytes"].as_u64().unwrap());
    let elsewhere = req
        .headers
        .iter()
        .filter(|(n, _)| !n.eq_ignore_ascii_case("authorization"))
        .any(|(_, v)| v.contains(token));
    assert!(
        !elsewhere && !req.target.contains(token),
        "the token outside Authorization"
    );
    // Nothing that answers the agent carries it either.
    let listed = format!("{:?}", e.jobs().list(&sc).await.unwrap());
    assert!(!listed.contains(token));
}

/// `keep` never waits for its callback: answered while the receiver sits on the attempt, and with
/// a slow `keep` the proof overtakes the answer.
#[tokio::test]
async fn keep_answers_without_waiting_and_the_proof_can_overtake_it() {
    let rx = CallbackReceiver::start().await;
    rx.script([Answer::Hang]);
    let e = Endpoint::start(MockOptions::notifying()).await;
    let (sc, id) = (scope("hang"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 120").await;
    let t = Instant::now();
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&rx.url))
        .await
        .unwrap();
    assert!(t.elapsed() < Duration::from_secs(1), "keep waited");
    assert_eq!(rx.wait_for(1, Duration::from_secs(2)).await.len(), 1);

    let rx = CallbackReceiver::start().await;
    let e = Endpoint::start(MockOptions {
        keep_delay: Duration::from_millis(800),
        ..MockOptions::notifying()
    })
    .await;
    let (sc, id) = (scope("slow"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 120").await;
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&rx.url))
        .await
        .unwrap();
    let answered = Instant::now();
    let got = rx.received();
    assert_eq!(got.len(), 1, "the proof did not overtake the slow keep");
    assert!(got[0].at < answered);
}

/// After the proof, a notification at the shell's exit with its group alive, and one at the end.
#[tokio::test]
async fn the_shell_exit_and_the_end_are_called_back() {
    let rx = CallbackReceiver::start().await;
    let e = Endpoint::start(MockOptions::notifying()).await;
    let (sc, id) = (scope("end"), JobId::fresh());
    let _hold = e
        .held(&sc, &id, "sleep 1; sleep 1 & echo bye; exit 3")
        .await;
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&rx.url))
        .await
        .unwrap();
    let got = rx.wait_for(3, Duration::from_secs(6)).await;
    assert_eq!(got.len(), 3, "want the proof, the shell's exit and the end");
    let gap = got[2].at.duration_since(got[1].at);
    assert!(
        gap >= Duration::from_millis(500) && gap <= Duration::from_secs(3),
        "the end's callback came {gap:?} after the shell exit's"
    );
    let listed = e.jobs().list(&sc).await.unwrap();
    assert_eq!(listed[0].status, JobStatus::Ended(3));
    // Nothing more once it is over.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(rx.received().len(), 3);
}

/// One attempt per job at a time, and a `2xx` covers only the changes from before its attempt
/// began: the job ends while the proof's attempt is in flight, so a new attempt starts as soon as
/// that one is answered, never beside it.
#[tokio::test]
async fn a_change_during_an_attempt_gets_an_attempt_of_its_own() {
    let rx = CallbackReceiver::start().await;
    rx.script([Answer::Slow(Duration::from_millis(1500), 200)]);
    let e = Endpoint::start(MockOptions::notifying()).await;
    let (sc, id) = (scope("overlap"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 0.5").await;
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&rx.url))
        .await
        .unwrap();
    let got = rx.wait_for(2, Duration::from_secs(5)).await;
    assert_eq!(
        got.len(),
        2,
        "the end during the proof's attempt was covered by its 2xx"
    );
    let gap = got[1].at.duration_since(got[0].at);
    assert!(
        gap >= Duration::from_millis(1400) && gap <= Duration::from_millis(2500),
        "the second attempt began {gap:?} after the first: want once the first was answered"
    );
    assert_eq!(
        e.jobs().list(&sc).await.unwrap()[0].status,
        JobStatus::Ended(0)
    );
    tokio::time::sleep(Duration::from_millis(1000)).await;
    assert_eq!(rx.received().len(), 2);
}

/// Over the wire: `503`, `429` and a connection error are retried on the schedule, `2xx` ends the
/// notification, `404` stops it.
#[tokio::test]
async fn the_mock_retries_on_the_schedule_and_stops() {
    let rx = CallbackReceiver::start().await;
    rx.script([
        Answer::Status(503),
        Answer::Status(429),
        Answer::Status(204),
    ]);
    let e = Endpoint::start(MockOptions::notifying()).await;
    let (sc, id) = (scope("retry"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 120").await;
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&rx.url))
        .await
        .unwrap();
    let got = rx.wait_for(3, Duration::from_secs(6)).await;
    assert_eq!(got.len(), 3);
    let tol = Duration::from_millis(300);
    for (i, (lo, hi)) in [(0.5, 1.0), (1.0, 2.0)].into_iter().enumerate() {
        let gap = got[i + 1].at.duration_since(got[i].at);
        assert!(
            gap + tol >= Duration::from_secs_f64(lo) && gap <= Duration::from_secs_f64(hi) + tol,
            "wait {} was {gap:?}, want {lo}–{hi} s",
            i + 1
        );
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(rx.received().len(), 3, "a 2xx did not end the notification");

    // A refusal that always will be stops it at once.
    let rx = CallbackReceiver::start().await;
    rx.script([Answer::Status(404)]);
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&rx.url))
        .await
        .unwrap();
    assert_eq!(rx.wait_for(1, Duration::from_secs(2)).await.len(), 1);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(rx.received().len(), 1, "a 404 was retried");

    // A connection error is retried: nothing listens on this port.
    let port = common::DeadPort::bind();
    let dead = format!("127.0.0.1:{}", port.port());
    e.jobs()
        .keep_notify(
            &sc,
            &id,
            1,
            "background",
            0,
            &notify_to(&format!("http://{dead}/x")),
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1300)).await;
    let tries = e
        .mock
        .callbacks()
        .into_iter()
        .filter(|a| a.url.contains(&dead))
        .count();
    assert!(tries >= 2, "a connection error was not retried ({tries})");
}

/// A `302` ends the notification: the token never goes where the redirect points.
#[tokio::test]
async fn a_redirect_is_never_followed() {
    let (a, b) = (
        CallbackReceiver::start().await,
        CallbackReceiver::start().await,
    );
    a.script([Answer::Redirect(b.url.clone())]);
    let e = Endpoint::start(MockOptions::notifying()).await;
    let (sc, id) = (scope("redirect"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 120").await;
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&a.url))
        .await
        .unwrap();
    assert_eq!(a.wait_for(1, Duration::from_secs(2)).await.len(), 1);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(a.received().len(), 1, "a 3xx was retried");
    assert!(b.received().is_empty(), "the redirect was followed");
}

/// Over `https` the server is verified: a certificate nobody trusts gets no request, and the
/// failed handshake is a connection error, retried.
#[tokio::test]
async fn the_server_is_verified_over_tls() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(
            vec![cert.cert.der().clone()],
            rustls::pki_types::PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into()),
        )
        .unwrap();
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = beyond_ai_test_support::ports::tokio_listener().await;
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(std::sync::Mutex::new((0usize, 0usize)));
    {
        let seen = seen.clone();
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                seen.lock().unwrap().0 += 1;
                if let Ok(mut tls) = acceptor.accept(sock).await {
                    let mut b = [0u8; 1];
                    if tls.read(&mut b).await.is_ok_and(|n| n > 0) {
                        seen.lock().unwrap().1 += 1;
                    }
                }
            }
        });
    }
    let e = Endpoint::start(MockOptions::notifying()).await;
    let (sc, id) = (scope("tls"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 120").await;
    e.jobs()
        .keep_notify(
            &sc,
            &id,
            1,
            "background",
            0,
            &notify_to(&format!(
                "https://localhost:{port}/_beyond/agent?session_id=x"
            )),
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1800)).await;
    let (connections, requests) = *seen.lock().unwrap();
    assert_eq!(requests, 0, "a request went to an unverified server");
    assert!(
        connections >= 2,
        "the failed handshake was not retried ({connections})"
    );
    assert!(e.mock.callbacks().iter().all(|a| a.status.is_none()));
}

/// The latest `keep` governs: a new one abandons the attempt in flight and its notification, and
/// one without `notify` stops everything; a removed job gets nothing more; a job kept without
/// `notify` is never called back.
#[tokio::test]
async fn callbacks_follow_the_latest_keep_and_stop_at_removal() {
    let e = Endpoint::start(MockOptions::notifying()).await;
    let (sc, id) = (scope("latest"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 120").await;
    let (a, b) = (
        CallbackReceiver::start().await,
        CallbackReceiver::start().await,
    );
    a.script([Answer::Hang]);
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&a.url))
        .await
        .unwrap();
    assert_eq!(a.wait_for(1, Duration::from_secs(2)).await.len(), 1);
    b.script(std::iter::repeat_n(Answer::Status(503), 20));
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&b.url))
        .await
        .unwrap();
    assert_eq!(b.wait_for(1, Duration::from_secs(2)).await.len(), 1);
    // A plain keep removes the notify: b's retries stop.
    e.jobs().keep(&sc, &id, 1, "background", 0).await.unwrap();
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert_eq!(a.received().len(), 1, "the abandoned notification went on");
    assert_eq!(
        b.received().len(),
        1,
        "a keep without notify left it in force"
    );

    // Removed: no attempt after it.
    let c = CallbackReceiver::start().await;
    c.script(std::iter::repeat_n(Answer::Status(503), 20));
    let other = JobId::fresh();
    e.jobs()
        .start_kept(&sc, &other, &spec("sleep 120"))
        .await
        .unwrap();
    e.jobs()
        .keep_notify(&sc, &other, 2, "background", 0, &notify_to(&c.url))
        .await
        .unwrap();
    assert_eq!(c.wait_for(1, Duration::from_secs(2)).await.len(), 1);
    e.jobs()
        .kill(&sc, std::slice::from_ref(&other), true)
        .await
        .unwrap();
    let after = c.received().len();
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert_eq!(c.received().len(), after, "an attempt after removal");

    // Kept without notify, it ends unannounced.
    let d = CallbackReceiver::start().await;
    let quiet = JobId::fresh();
    e.jobs()
        .start_kept(&sc, &quiet, &spec("sleep 0.3"))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(d.received().is_empty());
    assert!(
        e.mock
            .callbacks()
            .iter()
            .all(|x| x.id != quiet.as_str() && x.id != other.as_str() || x.url == c.url)
    );
}

/// The proof matrix's lost callbacks: the proof, the changes, or every one is never sent.
#[tokio::test]
async fn lost_callbacks_are_never_sent() {
    for (lose, want_proof, want_end) in [
        (LoseCallbacks::Proof, false, true),
        (LoseCallbacks::Changes, true, false),
        (LoseCallbacks::All, false, false),
    ] {
        let rx = CallbackReceiver::start().await;
        let e = Endpoint::start(MockOptions {
            lose_callbacks: lose,
            ..MockOptions::notifying()
        })
        .await;
        let (sc, id) = (scope("lose"), JobId::fresh());
        let _hold = e.held(&sc, &id, "sleep 1").await;
        e.jobs()
            .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&rx.url))
            .await
            .unwrap();
        let t = Instant::now();
        tokio::time::sleep(Duration::from_millis(400)).await;
        let proof = !rx.received().is_empty();
        tokio::time::sleep(Duration::from_millis(1600)).await;
        let total = rx.received().len();
        assert_eq!(proof, want_proof, "{lose:?}");
        assert_eq!(
            total,
            usize::from(want_proof) + usize::from(want_end),
            "{lose:?} after {:?}",
            t.elapsed()
        );
    }
}

/// Without egress every attempt is a connection error, and the endpoint does not advertise.
#[tokio::test]
async fn without_egress_no_callback_leaves_the_sandbox() {
    let rx = CallbackReceiver::start().await;
    let e = Endpoint::start(MockOptions {
        egress: false,
        ..MockOptions::notifying()
    })
    .await;
    assert!(!e.runner.job_offer().await.notify);
    let (sc, id) = (scope("dark"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 120").await;
    e.jobs()
        .keep_notify(&sc, &id, 1, "background", 0, &notify_to(&rx.url))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1300)).await;
    assert!(rx.received().is_empty());
    let attempts = e.mock.callbacks();
    assert!(attempts.len() >= 2 && attempts.iter().all(|a| a.status.is_none()));
}

/// An idle-killing sandbox kills a kept job once no request is open, and (deployed honestly) does
/// not advertise notify.
#[tokio::test]
async fn an_idle_killing_sandbox_kills_kept_jobs() {
    let e = Endpoint::start(MockOptions {
        idle_kill: Some(Duration::from_millis(300)),
        ..MockOptions::default()
    })
    .await;
    assert!(!e.runner.job_offer().await.notify);
    let (sc, id) = (scope("idle"), JobId::fresh());
    e.jobs()
        .start_kept(&sc, &id, &spec("sleep 120"))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(900)).await;
    let listed = e.jobs().list(&sc).await.unwrap();
    assert_eq!(listed[0].status, JobStatus::Ended(137), "{listed:?}");
}

/// The client: a refused `notify` is `InvalidInput`, so the caller can keep without it.
#[tokio::test]
async fn a_refused_notify_is_invalid_input_to_the_client() {
    let e = Endpoint::start(MockOptions::notifying()).await;
    let (sc, id) = (scope("refused"), JobId::fresh());
    let _hold = e.held(&sc, &id, "sleep 120").await;
    let err = e
        .jobs()
        .keep_notify(
            &sc,
            &id,
            1,
            "background",
            0,
            &notify_to("http://edge.example/x"),
        )
        .await
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
    let job = e.jobs().list(&sc).await.unwrap().remove(0);
    assert!(!job.kept);
    assert!(e.mock.callbacks().is_empty());
    e.jobs().keep(&sc, &id, 1, "background", 0).await.unwrap();
}
