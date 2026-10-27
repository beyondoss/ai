//! The session side of exec notify's job-end callbacks ("Sandbox job notify" in
//! [`crate::exec_endpoint`]): which sessions this process holds open, and the driver that feeds
//! each session's [`kernel`] its events and performs what it decides.
//!
//! Every decision is the kernel's ([`kernel::Kernel::step`], pure and deterministic). This module
//! only supplies what the kernel cannot have: the clock (as [`kernel::Millis`] since the session
//! registered), the table facts a callback needs ([`JobTracker::facts`]), the `list` a reconcile
//! runs ([`JobTracker::list`]), a timer for [`kernel::Kernel::next_wake`], and a reply path of its
//! own for every waiting callback.
//!
//! A callback never opens a session: [`NotifyHub::find`] only looks up a session a task in this
//! process already holds ([`SessionNotify::register`]).

pub mod kernel;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, Weak};

use futures::future::BoxFuture;
use tokio::sync::{Notify, oneshot};
use tokio::time::Instant;

use crate::exec_endpoint::jobs::JobId;
use crate::notify_token::NotifyKey;
use kernel::{Action, CallId, Event, Facts, Kernel, Listed, Millis, TableJob};

/// Check `--notify-origin`: an absolute `http`/`https` URL with a host and nothing past the origin
/// (no path but `/`, no query, no fragment, no userinfo).
pub fn check_origin(origin: &str) -> Result<(), String> {
    let url = url::Url::parse(origin)
        .map_err(|e| format!("--notify-origin {origin:?} is not a URL: {e}"))?;
    let bare = matches!(url.scheme(), "http" | "https")
        && url.host_str().is_some_and(|h| !h.is_empty())
        && url.username().is_empty()
        && url.password().is_none()
        && url.path() == "/"
        && url.query().is_none()
        && url.fragment().is_none();
    if bare {
        Ok(())
    } else {
        Err(format!(
            "--notify-origin {origin:?} must be an http or https origin, like https://edge.example"
        ))
    }
}

/// What the replica's metrics count an answer as.
fn callback_outcome(why: kernel::Why) -> crate::metrics::JobCallback {
    use crate::metrics::JobCallback;
    use kernel::Why;
    match why {
        Why::Proof => JobCallback::Proof,
        Why::Recorded => JobCallback::Duplicate,
        Why::Hint | Why::NotReadBack | Why::Listed => JobCallback::Accepted,
        Why::Unknown => JobCallback::Gone,
        Why::Taken | Why::Late | Why::Full | Why::Unlisted => JobCallback::Unavailable,
    }
}

fn held_reason(why: kernel::HeldWhy) -> crate::metrics::JobHeld {
    use crate::metrics::JobHeld;
    use kernel::HeldWhy;
    match why {
        HeldWhy::Unavailable => JobHeld::Unavailable,
        HeldWhy::Refused => JobHeld::Refused,
        HeldWhy::KeepFailed => JobHeld::KeepFailed,
        HeldWhy::NoProof => JobHeld::NoProof,
        HeldWhy::Ended => JobHeld::Ended,
    }
}

/// A session's job table, as the kernel's driver sees it.
pub trait JobTracker: Send + Sync {
    /// What today's job table says of the job a callback names.
    fn facts(&self, id: &JobId) -> Facts;

    /// The live reconcile's only request: `list` the scope. `None` when it could not be listed.
    fn list(&self) -> BoxFuture<'static, Option<BTreeMap<String, Listed>>>;

    /// Perform an action about a job ([`Action::SendKeep`], [`Action::CloseStream`],
    /// [`Action::ApplyShellExit`], [`Action::Follow`], [`Action::ReadInThenFollow`]).
    fn perform(&self, action: Action);
}

/// The sessions this process holds open, by tenant and session id, and the key their callbacks'
/// tokens are checked under. One per replica.
pub struct NotifyHub {
    key: NotifyKey,
    sessions: Mutex<HashMap<(String, String), Weak<SessionNotify>>>,
    metrics: Option<Arc<crate::metrics::Metrics>>,
}

impl NotifyHub {
    pub fn new(key: NotifyKey) -> Arc<Self> {
        Self::with_metrics(key, None)
    }

    /// [`Self::new`], counting callbacks and fallbacks in `metrics` (`--metrics-listen`).
    pub fn with_metrics(
        key: NotifyKey,
        metrics: Option<Arc<crate::metrics::Metrics>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            key,
            sessions: Mutex::new(HashMap::new()),
            metrics,
        })
    }

    /// The replica's metrics, when it publishes them.
    pub fn metrics(&self) -> Option<&Arc<crate::metrics::Metrics>> {
        self.metrics.as_ref()
    }

    /// The replica's notify key: the one it mints and verifies under.
    pub fn key(&self) -> &NotifyKey {
        &self.key
    }

    /// The session a task in this process holds for `tenant`, if any. Never opens one.
    pub fn find(&self, tenant: &str, session_id: &str) -> Option<Arc<SessionNotify>> {
        lock(&self.sessions)
            .get(&(tenant.to_string(), session_id.to_string()))
            .and_then(Weak::upgrade)
    }

    /// How many sessions are registered.
    pub fn len(&self) -> usize {
        lock(&self.sessions)
            .values()
            .filter(|w| w.strong_count() > 0)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct Inner {
    kernel: Kernel,
    /// Each waiting callback's own reply path.
    calls: HashMap<CallId, oneshot::Sender<u16>>,
    next_call: CallId,
}

/// One open session's notify state and its driver. Registered with its [`NotifyHub`] for as long
/// as the session task holds it; dropping it unregisters the session.
pub struct SessionNotify {
    hub: Weak<NotifyHub>,
    key: (String, String),
    /// Zero on the kernel's clock: an hour before the session registered.
    base: Instant,
    inner: Mutex<Inner>,
    tracker: Mutex<Option<Weak<dyn JobTracker>>>,
    /// Wakes the timer task to recompute [`Kernel::next_wake`].
    timer: Arc<Notify>,
    /// The origin the fleet's edge serves sessions at (`--notify-origin`): what a `keep`'s
    /// `notify.url` is built on. `None`: notify is not available to this session.
    origin: Mutex<Option<String>>,
}

impl SessionNotify {
    /// Register `session_id` of `tenant` as held by this process. A later registration of the
    /// same session replaces an earlier one.
    pub fn register(hub: &Arc<NotifyHub>, tenant: &str, session_id: &str) -> Arc<Self> {
        let key = (tenant.to_string(), session_id.to_string());
        let this = Arc::new(Self {
            hub: Arc::downgrade(hub),
            key: key.clone(),
            // An hour back, so an arrival read just before registering is still on the scale.
            base: Instant::now()
                .checked_sub(std::time::Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
            inner: Mutex::new(Inner {
                kernel: Kernel::new(),
                calls: HashMap::new(),
                next_call: 0,
            }),
            tracker: Mutex::new(None),
            timer: Arc::new(Notify::new()),
            origin: Mutex::new(None),
        });
        lock(&hub.sessions).insert(key, Arc::downgrade(&this));
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(timer(Arc::downgrade(&this), this.timer.clone()));
        }
        this
    }

    /// The job table callbacks are answered from. Until one is set, the session is treated as not
    /// having read its sandbox's jobs back.
    pub fn set_tracker(&self, tracker: Weak<dyn JobTracker>) {
        *lock(&self.tracker) = Some(tracker);
    }

    /// Make notify available to this session's jobs, at `origin` (the edge's, configured by the
    /// operator), or not (`None`).
    pub fn set_origin(&self, origin: Option<String>) {
        *lock(&self.origin) = origin;
    }

    /// The `notify` a `keep` of job `id` carries: the session's own address at the origin, and a
    /// token minted for this session and job. `None` when notify is not available.
    pub fn target_for(&self, id: &JobId) -> Option<crate::exec_endpoint::jobs::JobNotify> {
        let origin = lock(&self.origin).clone()?;
        let hub = self.hub.upgrade()?;
        let (tenant, session) = &self.key;
        let query = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("session_id", session)
            .append_pair("tenant", tenant)
            .finish();
        Some(crate::exec_endpoint::jobs::JobNotify {
            url: format!("{}/_beyond/agent?{query}", origin.trim_end_matches('/')),
            token: hub.key().mint(tenant, session, id),
        })
    }

    fn tracker(&self) -> Option<Arc<dyn JobTracker>> {
        lock(&self.tracker).as_ref().and_then(Weak::upgrade)
    }

    /// Now, on the kernel's clock.
    pub fn now(&self) -> Millis {
        self.millis(Instant::now())
    }

    fn millis(&self, at: Instant) -> Millis {
        u64::try_from(at.saturating_duration_since(self.base).as_millis()).unwrap_or(u64::MAX)
    }

    /// Feed the kernel one event and perform what it decides.
    pub fn feed(self: &Arc<Self>, event: Event) {
        for action in self.feed_collect(event) {
            self.perform_now(action);
        }
    }

    /// Feed the kernel one event: answer callbacks and start reconciles as it decides, and return
    /// its actions about jobs for the caller to perform (a `keep` it must await, say).
    pub fn feed_collect(self: &Arc<Self>, event: Event) -> Vec<Action> {
        let metrics = self.hub.upgrade().and_then(|h| h.metrics().cloned());
        let (actions, answers) = {
            let mut inner = lock(&self.inner);
            let actions = inner.kernel.step(event);
            let mut answers = Vec::new();
            let mut rest = Vec::new();
            for a in actions {
                match a {
                    Action::Answer { call, status, why } => {
                        if let Some(m) = &metrics {
                            m.job_callback(callback_outcome(why));
                        }
                        if let Some(tx) = inner.calls.remove(&call) {
                            answers.push((tx, status));
                        }
                    }
                    Action::Held { why, .. } => {
                        if let Some(m) = &metrics {
                            m.job_held(held_reason(why));
                        }
                    }
                    other => rest.push(other),
                }
            }
            (rest, answers)
        };
        for (tx, status) in answers {
            let _ = tx.send(status);
        }
        let mut rest = Vec::new();
        for action in actions {
            match action {
                Action::StartReconcile { .. } => self.perform_now(action),
                other => rest.push(other),
            }
        }
        self.timer.notify_one();
        rest
    }

    /// Perform one of the kernel's actions: a reconcile here, anything about a job through the
    /// job table.
    pub fn perform_now(self: &Arc<Self>, action: Action) {
        match action {
            Action::StartReconcile { reconcile } => {
                let me = Arc::downgrade(self);
                let list = self.tracker().map(|t| t.list());
                let run = async move {
                    let jobs = match list {
                        Some(list) => list.await,
                        None => None,
                    };
                    if let Some(me) = me.upgrade() {
                        let now = me.now();
                        me.feed(Event::Listed {
                            reconcile,
                            jobs,
                            now,
                        });
                    }
                };
                if let Ok(rt) = tokio::runtime::Handle::try_current() {
                    rt.spawn(run);
                }
            }
            Action::Answer { .. } | Action::Held { .. } => {}
            other => {
                if let Some(t) = self.tracker() {
                    t.perform(other);
                }
            }
        }
    }

    /// Answer a verified callback for job `id` whose head was read at `arrived`: the status to
    /// answer with (`200`, `410` or `503`). Dropping the future (its connection closed) stops the
    /// wait.
    pub async fn callback(self: &Arc<Self>, id: &JobId, arrived: Instant) -> u16 {
        let facts = self.tracker().map_or(
            Facts {
                open_reconcile_done: false,
                job: TableJob::Absent,
            },
            |t| t.facts(id),
        );
        let (tx, rx) = oneshot::channel();
        let call = {
            let mut inner = lock(&self.inner);
            let call = inner.next_call;
            inner.next_call += 1;
            inner.calls.insert(call, tx);
            call
        };
        let mut guard = Waiting {
            session: self.clone(),
            call,
            answered: false,
        };
        self.feed(Event::Callback {
            call,
            job: id.as_str().to_string(),
            facts,
            now: self.millis(arrived),
        });
        let status = rx.await.unwrap_or(503);
        guard.answered = true;
        status
    }

    /// Count how long after a job's end (`ended`, epoch seconds) the session recorded it.
    pub fn end_recorded(&self, ended: u64) {
        let Some(m) = self.hub.upgrade().and_then(|h| h.metrics().cloned()) else {
            return;
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        #[allow(clippy::cast_precision_loss)]
        m.job_end_to_notice_seconds
            .observe((now - ended as f64).max(0.0));
    }

    /// The kernel's state, for tests and diagnostics.
    pub fn kernel(&self) -> Kernel {
        lock(&self.inner).kernel.clone()
    }
}

/// A callback still waiting: dropped before its answer (the connection closed), it stops waiting.
struct Waiting {
    session: Arc<SessionNotify>,
    call: CallId,
    answered: bool,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if self.answered {
            return;
        }
        lock(&self.session.inner).calls.remove(&self.call);
        self.session.feed(Event::CallClosed { call: self.call });
    }
}

impl Drop for SessionNotify {
    fn drop(&mut self) {
        self.timer.notify_one();
        if let Some(hub) = self.hub.upgrade() {
            let mut sessions = lock(&hub.sessions);
            if sessions
                .get(&self.key)
                .is_some_and(|w| w.strong_count() == 0)
            {
                sessions.remove(&self.key);
            }
        }
    }
}

/// Feed [`Event::Tick`] whenever the kernel has something due. Holds no strong reference while
/// it waits, so the session can go away.
async fn timer(session: Weak<SessionNotify>, wake: Arc<Notify>) {
    loop {
        let notified = wake.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let due = {
            let Some(s) = session.upgrade() else { return };
            let next = lock(&s.inner).kernel.next_wake();
            next.map(|t| s.base + std::time::Duration::from_millis(t))
        };
        match due {
            Some(at) => {
                tokio::select! {
                    () = tokio::time::sleep_until(at) => {}
                    () = &mut notified => continue,
                }
            }
            None => {
                notified.await;
                continue;
            }
        }
        let Some(s) = session.upgrade() else { return };
        let now = s.now();
        s.feed(Event::Tick { now });
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use kernel::JobState;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    struct Fake {
        facts: Facts,
        lists: AtomicU32,
        listing: Option<Listed>,
        hold: Duration,
        performed: Mutex<Vec<Action>>,
    }

    impl JobTracker for Fake {
        fn facts(&self, _id: &JobId) -> Facts {
            self.facts
        }
        fn list(&self) -> BoxFuture<'static, Option<BTreeMap<String, Listed>>> {
            self.lists.fetch_add(1, Ordering::SeqCst);
            let (l, hold) = (self.listing, self.hold);
            Box::pin(async move {
                tokio::time::sleep(hold).await;
                l.map(|l| BTreeMap::from([(ID.to_string(), l)]))
            })
        }
        fn perform(&self, action: Action) {
            lock(&self.performed).push(action);
        }
    }

    const ID: &str = "0123456789abcdef";

    fn fake(job: TableJob, listing: Option<Listed>) -> Arc<Fake> {
        Arc::new(Fake {
            facts: Facts {
                open_reconcile_done: true,
                job,
            },
            lists: AtomicU32::new(0),
            listing,
            hold: Duration::from_millis(30),
            performed: Mutex::new(Vec::new()),
        })
    }

    fn hub() -> Arc<NotifyHub> {
        NotifyHub::new(NotifyKey::from_secret_bytes(&[7; 32]))
    }

    fn id() -> JobId {
        JobId::parse(ID).unwrap()
    }

    fn session(hub: &Arc<NotifyHub>, fake: &Arc<Fake>) -> Arc<SessionNotify> {
        let s = SessionNotify::register(hub, "acme", "s1");
        let t: Arc<dyn JobTracker> = fake.clone();
        s.set_tracker(Arc::downgrade(&t));
        s
    }

    /// Make the job unheld through the release row, as production does.
    fn release(s: &Arc<SessionNotify>) {
        s.feed(Event::Kept {
            job: ID.into(),
            notify_available: true,
            adopted: false,
        });
        let now = s.now();
        s.feed(Event::KeepAnswered {
            job: ID.into(),
            outcome: kernel::KeepOutcome::Succeeded,
            now,
        });
    }

    /// The value of a series in a scrape.
    fn sample(text: &str, series: &str) -> f64 {
        text.lines()
            .find_map(|l| {
                l.strip_prefix(series)
                    .map(|v| v.trim().parse::<f64>().unwrap())
            })
            .unwrap_or_else(|| panic!("no {series} in {text}"))
    }

    #[tokio::test]
    async fn every_counter_moves() {
        let metrics = crate::metrics::Metrics::new().unwrap();
        let hub = NotifyHub::with_metrics(
            NotifyKey::from_secret_bytes(&[7; 32]),
            Some(metrics.clone()),
        );
        let scrape = || String::from_utf8(metrics.encode()).unwrap();
        let cb = |o: &str| format!("agent_job_callbacks_total{{outcome=\"{o}\"}}");
        let held = |r: &str| format!("agent_job_notify_held_total{{reason=\"{r}\"}}");
        for o in ["accepted", "proof", "duplicate", "gone", "unavailable"] {
            assert_eq!(sample(&scrape(), &cb(o)), 0.0, "{o}");
        }

        // Callback answers, by the table's facts.
        for (job, outcome) in [
            (TableJob::Running, "accepted"),
            (TableJob::Ended, "duplicate"),
            (TableJob::Absent, "gone"),
            (TableJob::Taken, "unavailable"),
        ] {
            let f = fake(job, Some(Listed::Running));
            let s = session(&hub, &f);
            s.callback(&id(), Instant::now()).await;
            assert_eq!(sample(&scrape(), &cb(outcome)), 1.0, "{outcome}");
        }
        // The proof.
        let f = fake(TableJob::Running, Some(Listed::Running));
        let s = session(&hub, &f);
        release(&s);
        assert_eq!(s.callback(&id(), Instant::now()).await, 200);
        assert_eq!(sample(&scrape(), &cb("proof")), 1.0);

        // Why a job stays held.
        let job = || ID.to_string();
        s.feed(Event::Kept {
            job: job(),
            notify_available: false,
            adopted: false,
        });
        assert_eq!(sample(&scrape(), &held("unavailable")), 1.0);
        for (outcome, reason) in [
            (kernel::KeepOutcome::Refused, "refused"),
            (kernel::KeepOutcome::Failed, "keep_failed"),
        ] {
            s.feed(Event::Kept {
                job: job(),
                notify_available: true,
                adopted: false,
            });
            let now = s.now();
            s.feed(Event::KeepAnswered {
                job: job(),
                outcome,
                now,
            });
            assert_eq!(sample(&scrape(), &held(reason)), 1.0, "{reason}");
        }
        s.feed(Event::Kept {
            job: job(),
            notify_available: true,
            adopted: false,
        });
        s.feed(Event::StreamEnded { job: job() });
        assert_eq!(sample(&scrape(), &held("ended")), 1.0);
        s.feed(Event::Kept {
            job: job(),
            notify_available: true,
            adopted: false,
        });
        let now = s.now();
        s.feed(Event::KeepAnswered {
            job: job(),
            outcome: kernel::KeepOutcome::Succeeded,
            now,
        });
        s.feed(Event::Tick {
            now: now + kernel::PROOF_WINDOW,
        });
        assert_eq!(sample(&scrape(), &held("no_proof")), 1.0);

        // From an end to its record.
        let ended = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 5;
        s.end_recorded(ended);
        assert_eq!(
            sample(&scrape(), "agent_job_end_to_notice_seconds_count"),
            1.0
        );
        assert!(sample(&scrape(), "agent_job_end_to_notice_seconds_sum") >= 4.0);
    }

    #[tokio::test]
    async fn a_session_is_found_only_while_held() {
        let hub = hub();
        assert!(hub.find("acme", "s1").is_none());
        let s = SessionNotify::register(&hub, "acme", "s1");
        assert!(hub.find("acme", "s1").is_some());
        assert!(hub.find("globex", "s1").is_none(), "another tenant");
        assert!(hub.find("acme", "s2").is_none());
        drop(s);
        assert!(hub.find("acme", "s1").is_none());
        assert!(hub.is_empty());
    }

    #[tokio::test]
    async fn answers_without_a_reconcile_follow_the_table() {
        let hub = hub();
        for (job, want) in [
            (TableJob::Running, 200),
            (TableJob::Ended, 200),
            (TableJob::Taken, 503),
            (TableJob::Absent, 410),
        ] {
            let f = fake(job, Some(Listed::Running));
            let s = session(&hub, &f);
            assert_eq!(s.callback(&id(), Instant::now()).await, want, "{job:?}");
            assert_eq!(f.lists.load(Ordering::SeqCst), 0);
        }
        // No tracker yet: not read back, so 200.
        let s = SessionNotify::register(&hub, "acme", "s9");
        assert_eq!(s.callback(&id(), Instant::now()).await, 200);
    }

    #[tokio::test]
    async fn the_proof_releases_and_a_later_callback_waits_for_the_list() {
        let hub = hub();
        let f = fake(TableJob::Running, Some(Listed::Running));
        let s = session(&hub, &f);
        release(&s);
        assert_eq!(s.callback(&id(), Instant::now()).await, 200);
        assert!(lock(&f.performed).contains(&Action::CloseStream { job: ID.into() }));
        let now = s.now();
        s.feed(Event::StreamClosed {
            job: ID.into(),
            offset: 5,
            shell_exit_seen: false,
            now,
        });
        assert!(matches!(
            s.kernel().job(ID),
            Some(JobState::Unheld { offset: 5, .. })
        ));
        // The release's own reconcile, then (a second later) the callback's.
        let t0 = Instant::now();
        assert_eq!(s.callback(&id(), Instant::now()).await, 200);
        assert_eq!(f.lists.load(Ordering::SeqCst), 2);
        assert!(
            t0.elapsed() >= Duration::from_millis(900),
            "{:?}",
            t0.elapsed()
        );
    }

    #[tokio::test]
    async fn a_list_that_fails_is_503_and_follows() {
        let hub = hub();
        let f = fake(TableJob::Running, None);
        let s = session(&hub, &f);
        release(&s);
        s.callback(&id(), Instant::now()).await;
        let now = s.now();
        s.feed(Event::StreamClosed {
            job: ID.into(),
            offset: 5,
            shell_exit_seen: false,
            now,
        });
        // The release's reconcile could not list: the job is followed, so held again, and a
        // callback now is answered from the table.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(lock(&f.performed).contains(&Action::Follow {
            job: ID.into(),
            from: 5,
            end: None
        }));
        assert_eq!(s.kernel().job(ID), None);
    }

    #[tokio::test]
    async fn a_closed_connection_stops_waiting() {
        let hub = hub();
        let mut f = fake(TableJob::Running, Some(Listed::Running));
        Arc::get_mut(&mut f).unwrap().hold = Duration::from_secs(5);
        let s = session(&hub, &f);
        release(&s);
        s.callback(&id(), Instant::now()).await;
        let now = s.now();
        s.feed(Event::StreamClosed {
            job: ID.into(),
            offset: 0,
            shell_exit_seen: false,
            now,
        });
        let job = id();
        let wait = s.callback(&job, Instant::now());
        let _ = tokio::time::timeout(Duration::from_millis(50), wait).await;
        assert_eq!(s.kernel().waiting(), 0);
    }
}
