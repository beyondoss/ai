//! Where a callback ends up, across real replicas: routing (E3), the token kept out of every log
//! (E7), and a job's end through a takeover, a ring change and a session reaped before it.

use super::*;

/// **E3** — a callback is routed as any request for its session id: to the hashed target first,
/// then, while it answers `503`, retried and the ring walked, until the replica that has the session
/// open answers. One session open on its hashed target is reached at once; one open elsewhere (it
/// opened while its hashed target was out of the ring, which then came back) is found by the walk —
/// the proof that releases its job and the end that follows both.
pub async fn job_ended_routed_as_any_request(kind: Kind, history: &Path) -> Outcome {
    let mut tg = tags("route");
    tg.push("routed".into());
    let (mut fleet, sandboxes, flags) = match notify_fleet(kind, history, 3, None, &tg).await {
        Ok(x) => x,
        Err(e) if e.starts_with(ATTACHED_SKIP) => {
            return Outcome::Skipped(e.trim_start_matches(ATTACHED_SKIP).to_owned());
        }
        Err(e) => return Outcome::failed_with(e),
    };
    let findings = per_sandbox(&mut fleet, &sandboxes, &flags, "E3", async |fleet, exec, flags, _, i| {
        let tag = format!("route{i}");
        let all = targets_named(fleet, &["r1", "r2", "r3"]);
        // Away from its hashed target: r1 is out of the ring while the session opens, so it lands
        // where the ring without r1 puts it; then r1 comes back and is the hash again.
        let session = (0..100_000)
            .map(|n| format!("s1.{tag}-{n}"))
            .find(|id| {
                ring_order(&all, id).first().is_some_and(|t| t.name == "r1")
                    && ring_order(&targets_named(fleet, &["r2", "r3"]), id)
                        .first()
                        .is_some_and(|t| t.name == "r2")
            })
            .ok_or("no session id hashing to r1, and to r2 without it")?;
        let grant = fleet.edge.grant("t1", &session, "s1", "/", &exec.url);
        fleet.retarget_excluding("r1");
        let placed = crate::edge::place(&fleet.edge, &session, &grant, Duration::from_secs(30)).await;
        let Placement::Served { addr, .. } = placed else {
            return Err(format!("could not place {session}: {placed:?}"));
        };
        let owner = name_at(fleet, &addr);
        let mut ws = workload::connect(&addr, &session, &grant).await?;
        retarget_all(fleet);
        workload::prompt(&mut ws, &format!("start-job-{tag}")).await?;
        let proof = nth_answer(fleet, &session, 200, 1, Duration::from_secs(20)).await?;
        flags.raise(&tag)?;
        let end = nth_answer(fleet, &session, 200, 2, Duration::from_secs(30)).await?;
        let n = notices_settled(&mut ws, Duration::from_secs(30)).await?;
        let walked = |e: &Value| {
            let a = attempts(e);
            a.first().is_some_and(|(r, s)| r == "r1" && s == "503")
                && a.iter().filter(|(r, _)| r == "r1").count() >= 2
                && a.last().is_some_and(|(r, s)| *r == owner && s == "200")
        };
        Ok(vec![
            finding("E3", owner == "r2", format!("{session} opened on {owner} while r1 was out of the ring")),
            finding(
                "E3",
                walked(&proof),
                format!("the proof went to the hashed target r1 first, was retried there, then walked to {owner}: {}", proof["attempts"]),
            ),
            finding(
                "E3",
                walked(&end),
                format!("so did the end: {}", end["attempts"]),
            ),
            finding("E3", n == 1, format!("the end was reported {n} time(s), want 1")),
        ])
    })
    .await;
    // And a session on its hashed target is reached at once, by the first attempt.
    let mut all = findings;
    match routed_directly(&fleet, &sandboxes[0].1, &flags).await {
        Ok(f) => all.extend(f),
        Err(e) => all.push(finding("E3", false, format!("on its hashed target: {e}"))),
    }
    report("job-ended-routed-as-any-request", &all)
}

async fn routed_directly(
    fleet: &Fleet,
    exec: &ExecMock,
    flags: &Flags,
) -> Result<Vec<Finding>, String> {
    let tag = "routed";
    let all = targets_named(fleet, &["r1", "r2", "r3"]);
    let session = session_on(&all, "r3", tag)?;
    let grant = fleet.edge.grant("t1", &session, "s1", "/", &exec.url);
    let (addr, mut ws) = open_and_start(fleet, &session, &grant, tag).await?;
    let proof = nth_answer(fleet, &session, 200, 1, Duration::from_secs(20)).await?;
    flags.raise(tag)?;
    let end = nth_answer(fleet, &session, 200, 2, Duration::from_secs(30)).await?;
    let n = notices_settled(&mut ws, Duration::from_secs(30)).await?;
    let first_try = |e: &Value| attempts(e).len() == 1 && e["by"] == "r3";
    Ok(vec![
        finding(
            "E3",
            name_at(fleet, &addr) == "r3" && first_try(&proof) && first_try(&end),
            format!(
                "native: a session open on its hashed target r3 was reached by the first attempt, \
                 proof {} and end {}",
                proof["attempts"], end["attempts"]
            ),
        ),
        finding(
            "E3",
            n == 1,
            format!("native: and the end was reported {n} time(s)"),
        ),
    ])
}

/// **E7** — the front never records a callback's `Authorization`. Real callbacks (proofs and ends,
/// from both sandboxes) pass through it with logging on; afterwards no token, nor any `bjn_v1.`
/// string, is in its log file, its in-memory log, or the run's history. The replicas' own output is
/// checked too, though that is the replica's rule (N20), not the edge's.
pub async fn job_ended_token_never_logged(kind: Kind, history: &Path) -> Outcome {
    let tg = tags("log");
    let (mut fleet, sandboxes, flags) = match notify_fleet(kind, history, 2, None, &tg).await {
        Ok(x) => x,
        Err(e) if e.starts_with(ATTACHED_SKIP) => {
            return Outcome::Skipped(e.trim_start_matches(ATTACHED_SKIP).to_owned());
        }
        Err(e) => return Outcome::failed_with(e),
    };
    let mut findings = per_sandbox(
        &mut fleet,
        &sandboxes,
        &flags,
        "E7",
        async |fleet, exec, flags, _, i| {
            let tag = format!("log{i}");
            let session = format!("s1.{tag}");
            let grant = fleet.edge.grant("t1", &session, "s1", "/", &exec.url);
            let (_, mut ws) = open_and_start(fleet, &session, &grant, &tag).await?;
            nth_answer(fleet, &session, 200, 1, Duration::from_secs(20)).await?;
            flags.raise(&tag)?;
            nth_answer(fleet, &session, 200, 2, Duration::from_secs(30)).await?;
            let n = notices_settled(&mut ws, Duration::from_secs(30)).await?;
            Ok(vec![finding(
                "E7",
                n == 1,
                format!("two callbacks passed through the front; the end was reported {n} time(s)"),
            )])
        },
    )
    .await;
    // The tokens the native sandbox was given.
    let tokens: Vec<String> = sandboxes[0]
        .1
        .bodies()
        .iter()
        .filter_map(|b| b["job"]["notify"]["token"].as_str().map(str::to_owned))
        .collect();
    findings.push(finding(
        "E7",
        !tokens.is_empty(),
        format!(
            "the native sandbox was given {} token(s) to call back with",
            tokens.len()
        ),
    ));
    let log_path = history.with_extension("front.jsonl");
    let file = std::fs::read_to_string(&log_path).unwrap_or_default();
    let memory = front_of(&fleet)
        .map(|f| Value::Array(f.log()).to_string())
        .unwrap_or_default();
    let hist = std::fs::read_to_string(history).unwrap_or_default();
    let entries = file.lines().filter(|l| l.contains("\"callback\"")).count();
    findings.push(finding(
        "E7",
        entries >= 4,
        format!(
            "the front logged {entries} callback(s) to {}",
            log_path.display()
        ),
    ));
    for (what, text) in [
        ("the front's log file", &file),
        ("the front's in-memory log", &memory),
        ("the run's history", &hist),
    ] {
        let leaked = text.contains("bjn_v1.") || tokens.iter().any(|t| text.contains(t.as_str()));
        findings.push(finding("E7", !leaked, format!("no token in {what}")));
    }
    for r in &fleet.replicas {
        let said = r.said();
        findings.push(finding(
            "N20",
            !said.contains("bjn_v1.") && !tokens.iter().any(|t| said.contains(t.as_str())),
            format!("no token in replica {}'s output", r.name),
        ));
    }
    report("job-ended-token-never-logged", &findings)
}

/// **Takeover** — a job kept and released on replica A, whose session then moves. A is killed; the
/// session takes over on B, whose first use of the sandbox adopts the job and keeps it again, with
/// notify, proved anew through the front. A comes back (restarted, empty, and the session's hashed
/// target again); the job ends; the callback reaches the restarted A first, is answered `503`, and
/// walks on to B, which reports the end exactly once. A late copy of the callback after that is
/// answered by B too, and reports nothing again.
pub async fn job_ended_after_takeover(kind: Kind, history: &Path) -> Outcome {
    let tg = tags("take");
    let (mut fleet, sandboxes, flags) = match notify_fleet(kind, history, 2, None, &tg).await {
        Ok(x) => x,
        Err(e) if e.starts_with(ATTACHED_SKIP) => {
            return Outcome::Skipped(e.trim_start_matches(ATTACHED_SKIP).to_owned());
        }
        Err(e) => return Outcome::failed_with(e),
    };
    let findings = per_sandbox(
        &mut fleet,
        &sandboxes,
        &flags,
        "takeover",
        async |fleet, exec, flags, _, i| {
            let tag = format!("take{i}");
            let all = targets_named(fleet, &["r1", "r2"]);
            let session = session_on(&all, "r1", &tag)?;
            let grant = fleet.edge.grant("t1", &session, "s1", "/", &exec.url);
            let (addr, ws) = open_and_start(fleet, &session, &grant, &tag).await?;
            let a = name_at(fleet, &addr);
            if a != "r1" {
                return Err(format!("{session} opened on {a}, not its hashed target r1"));
            }
            let proof = nth_answer(fleet, &session, 200, 1, Duration::from_secs(20)).await?;
            let native = SANDBOXES[i].1;
            let keeps_before = native.then(|| native_keep_bodies(exec, &session).len());

            // A dies outright; the session moves to B.
            drop(ws);
            let a_idx = replica_at(fleet, &addr).ok_or("A matches no replica")?;
            fleet.kill_hard(a_idx)?;
            fleet
                .history
                .record("owner_killed", json!({ "replica": a, "session": session }));
            fleet.retarget_excluding("r1");
            let began = Instant::now();
            let placed =
                crate::edge::place(&fleet.edge, &session, &grant, crate::edge::RETRY_BUDGET).await;
            let Placement::Served { addr: b_addr, .. } = placed else {
                return Err(format!("no replica took {session} over: {placed:?}"));
            };
            let took_over = began.elapsed();
            let b = name_at(fleet, &b_addr);
            let mut ws = workload::connect(&b_addr, &session, &grant).await?;
            // B's first use of the sandbox runs its open reconcile, which adopts the job and keeps it.
            workload::command(
                &mut ws,
                json!({ "type": "bash", "command": "true" }),
                "bash",
            )
            .await?;
            let reproof = nth_answer(fleet, &session, 200, 2, Duration::from_secs(20)).await?;
            let keeps_after = native.then(|| native_keep_bodies(exec, &session).len());

            // A comes back, empty, and is the session's hashed target again.
            fleet.restart(a_idx).await?;
            retarget_all(fleet);
            flags.raise(&tag)?;
            let end = nth_answer(fleet, &session, 200, 3, Duration::from_secs(30)).await?;
            let n = notices_settled(&mut ws, Duration::from_secs(30)).await?;

            let mut out = vec![
                finding(
                    "takeover",
                    proof["by"] == "r1",
                    format!(
                        "kept and released on A ({a}): the proof was answered by {}",
                        proof["by"]
                    ),
                ),
                finding(
                    "takeover",
                    b == "r2",
                    format!("B ({b}) took the session over after {took_over:.1?}"),
                ),
                finding(
                    "takeover",
                    reproof["by"] == "r2",
                    format!(
                        "B kept the job again and its proof was answered by {}",
                        reproof["by"]
                    ),
                ),
            ];
            if let (Some(before), Some(after)) = (keeps_before, keeps_after) {
                out.push(finding(
                    "takeover",
                    after > before,
                    format!(
                        "the sandbox saw {} keep(s) with notify before the takeover, {after} after",
                        before
                    ),
                ));
            }
            let a_then_b = {
                let at = attempts(&end);
                at.first().is_some_and(|(r, s)| r == "r1" && s == "503")
                    && at.last().is_some_and(|(r, s)| r == "r2" && s == "200")
            };
            out.push(finding(
                "takeover",
                a_then_b,
                format!(
                    "the end's callback reached the restarted A first (503) and walked on to B: {}",
                    end["attempts"]
                ),
            ));
            out.push(finding(
                "takeover",
                n == 1,
                format!("B reported the end {n} time(s), want 1"),
            ));

            // A late copy, replayed with the token the sandbox holds (native only: the baseline's is in
            // its supervisor's 0600 file). Still A first, still B's to answer, and nothing new.
            if let Some(token) = native_token(exec, &session) {
                let scope_id = native_scope_id(exec, &session);
                let body = scope_id
                    .map(|(s, id)| format!(r#"{{"type":"job_ended","scope":"{s}","id":"{id}"}}"#))
                    .unwrap_or_default();
                let raw = request(
                    "POST",
                    &agent_path(&session, ""),
                    &[bearer(&token), "Content-Type: application/json".into()],
                    &body,
                );
                let front = front_of(fleet)?;
                let (answer, _) = send(front.authority(), &raw).await?;
                let late = answers(fleet, &session).pop().unwrap_or_default();
                let n_after = notices(&mut ws).await?;
                out.push(finding(
                "takeover",
                front::status_of(&answer) == 200
                    && attempts(&late).first().is_some_and(|(r, s)| r == "r1" && s == "503")
                    && late["by"] == "r2",
                format!(
                    "a late callback reached the restarted A (503) and was answered {} by {}: {}",
                    front::status_of(&answer),
                    late["by"],
                    late["attempts"]
                ),
            ));
                out.push(finding(
                    "takeover",
                    n_after == 1,
                    format!("and the end is still reported {n_after} time(s)"),
                ));
            }
            Ok(out)
        },
    )
    .await;
    report("job-ended-after-takeover", &findings)
}

/// **Ring change** — replicas leave and rejoin the ring while a job runs, and its end still reaches
/// the replica that has the session open, once. First a replica that is not the owner leaves and
/// rejoins (a deploy): the session's hash does not move, and nothing changes. Then the owner itself
/// is out of the ring when the job ends: every callback walks the rest of the ring and is answered
/// `503`, the sandbox keeps retrying, and the first retry after the owner rejoins reaches it.
pub async fn job_ended_across_a_ring_change(kind: Kind, history: &Path) -> Outcome {
    let tg = tags("ring");
    let (mut fleet, sandboxes, flags) = match notify_fleet(kind, history, 3, None, &tg).await {
        Ok(x) => x,
        Err(e) if e.starts_with(ATTACHED_SKIP) => {
            return Outcome::Skipped(e.trim_start_matches(ATTACHED_SKIP).to_owned());
        }
        Err(e) => return Outcome::failed_with(e),
    };
    let findings = per_sandbox(&mut fleet, &sandboxes, &flags, "ring", async |fleet, exec, flags, _, i| {
        let tag = format!("ring{i}");
        let all = targets_named(fleet, &["r1", "r2", "r3"]);
        let session = session_on(&all, "r1", &tag)?;
        let grant = fleet.edge.grant("t1", &session, "s1", "/", &exec.url);
        let (addr, mut ws) = open_and_start(fleet, &session, &grant, &tag).await?;
        let owner = name_at(fleet, &addr);
        let proof = nth_answer(fleet, &session, 200, 1, Duration::from_secs(20)).await?;

        // A non-owner is deployed: it leaves, then rejoins on a new port.
        let r2 = fleet.replicas.iter().position(|r| r.name == "r2").ok_or("no r2")?;
        fleet.stop_gracefully(r2, Duration::from_secs(60))?;
        fleet.retarget_excluding("r2");
        let hash_while_away = fleet.edge.route(&session).map(|t| t.name.clone());
        fleet.restart(r2).await?;
        retarget_all(fleet);

        // The owner leaves the ring, still serving its session, and the job ends.
        let before_503 = count_answered(fleet, &session, 503);
        fleet.retarget_excluding(&owner);
        flags.raise(&tag)?;
        let away = nth_answer(fleet, &session, 503, before_503 + 1, Duration::from_secs(45)).await?;
        // The owner rejoins; the sandbox's next attempt finds it.
        retarget_all(fleet);
        let end = nth_answer(fleet, &session, 200, 2, Duration::from_secs(45)).await?;
        let n = notices_settled(&mut ws, Duration::from_secs(30)).await?;
        let others: Vec<String> = attempts(&away).into_iter().map(|(r, _)| r).collect();
        Ok(vec![
            finding("ring", owner == "r1" && proof["by"] == "r1", format!("released on its hashed target, {owner}")),
            finding(
                "ring",
                hash_while_away.as_deref() == Some("r1"),
                format!("with r2 out of the ring the session still hashed to {hash_while_away:?}"),
            ),
            finding(
                "ring",
                !others.is_empty() && !others.contains(&owner) && away["answered"] == 503,
                format!("with the owner out of the ring the end's callback walked {others:?} and was answered 503"),
            ),
            finding(
                "ring",
                end["by"] == owner.as_str() && attempts(&end).len() == 1,
                format!("after the owner rejoined, the sandbox's retry reached it first: {}", end["attempts"]),
            ),
            finding("ring", n == 1, format!("the end was reported {n} time(s), want 1")),
        ])
    })
    .await;
    report("job-ended-across-a-ring-change", &findings)
}

/// **Closed session** — the session is reaped before its job ends. Every callback then walks the
/// whole ring and is answered `503` by every replica, and none of them opens the session: its
/// directory is untouched. When the session reopens and uses its sandbox, it reads the end back and
/// reports it once; the sandbox's retries that follow change nothing.
pub async fn job_ended_after_the_session_closed(kind: Kind, history: &Path) -> Outcome {
    let tg = tags("closed");
    let (mut fleet, sandboxes, flags) = match notify_fleet(kind, history, 2, Some(1), &tg).await {
        Ok(x) => x,
        Err(e) if e.starts_with(ATTACHED_SKIP) => {
            return Outcome::Skipped(e.trim_start_matches(ATTACHED_SKIP).to_owned());
        }
        Err(e) => return Outcome::failed_with(e),
    };
    let findings = per_sandbox(&mut fleet, &sandboxes, &flags, "closed", async |fleet, exec, flags, _, i| {
        let tag = format!("closed{i}");
        let session = format!("s1.{tag}");
        let grant = fleet.edge.grant("t1", &session, "s1", "/", &exec.url);
        let (addr, ws) = open_and_start(fleet, &session, &grant, &tag).await?;
        let owner = name_at(fleet, &addr);
        nth_answer(fleet, &session, 200, 1, Duration::from_secs(20)).await?;
        drop(ws);
        // Reaped: detached for longer than the 1 s idle timeout.
        tokio::time::sleep(Duration::from_secs(4)).await;
        let dir = fleet
            .substrate
            .session_dir("s1", "t1", &session)
            .ok_or("the session wrote no directory")?;
        let before = snapshot(&dir);

        flags.raise(&tag)?;
        let walked = nth_answer(fleet, &session, 503, 1, Duration::from_secs(45)).await?;
        let reached: std::collections::BTreeSet<String> = attempts(&walked)
            .into_iter()
            .filter(|(_, s)| s == "503")
            .map(|(r, _)| r)
            .collect();
        let untouched = snapshot(&dir) == before;
        let opened_by_callback = count_answered(fleet, &session, 200) > 1;

        // Reopen and use the sandbox: the end is read back and reported once.
        let entries_before = answers(fleet, &session).len();
        let ops_before = exec.bodies().len();
        let placed = crate::edge::place(&fleet.edge, &session, &grant, Duration::from_secs(30)).await;
        let Placement::Served { addr, .. } = placed else {
            return Err(format!("could not reopen {session}: {placed:?}"));
        };
        let mut ws = workload::connect(&addr, &session, &grant).await?;
        workload::prompt(&mut ws, &format!("check-{tag}")).await?;
        let n = notices_settled(&mut ws, Duration::from_secs(30)).await?;
        // Whatever the sandbox still sends must not report the end again. Once the reopened
        // session has read the end back it removes the job, and an endpoint makes no attempt for a
        // removed job (N13), so there may be none at all.
        tokio::time::sleep(Duration::from_secs(8)).await;
        let n_later = notices(&mut ws).await?;
        let after: Vec<Value> = answers(fleet, &session).split_off(entries_before);
        let removed = exec.bodies()[ops_before..]
            .iter()
            .any(|b| matches!(b["job"]["op"].as_str(), Some("remove" | "wipe")));
        Ok(vec![
            finding("closed", owner == "r1" || owner == "r2", format!("released on {owner}, then reaped")),
            finding(
                "closed",
                reached.len() == fleet.replicas.len() && walked["answered"] == 503,
                format!("the end's callback walked the whole ring and every replica answered 503: {}", walked["attempts"]),
            ),
            finding("closed", untouched && !opened_by_callback, "no callback opened the session: its directory is untouched"),
            finding("closed", n == 1, format!("on reopen the end was reported {n} time(s), want 1")),
            finding(
                "closed",
                n_later == 1,
                format!(
                    "8 s on, still {n_later} time(s); the sandbox sent {} more callback(s) ({}){}",
                    after.len(),
                    summarize(&after),
                    if removed { ", the session having removed the job" } else { "" }
                ),
            ),
        ])
    })
    .await;
    report("job-ended-after-the-session-closed", &findings)
}

/// Every file under `dir` with its length and modification time.
fn snapshot(dir: &Path) -> Vec<(PathBuf, u64, Option<std::time::SystemTime>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            match e.metadata() {
                Ok(m) if m.is_dir() => stack.push(p),
                Ok(m) => out.push((p, m.len(), m.modified().ok())),
                Err(_) => {}
            }
        }
    }
    out.sort();
    out
}

/// The `keep` requests with notify a native sandbox received for `session`.
fn native_keep_bodies(exec: &ExecMock, session: &str) -> Vec<Value> {
    let needle = format!("session_id={session}&");
    exec.bodies()
        .into_iter()
        .filter(|b| b["job"]["op"] == "keep")
        .filter(|b| {
            b["job"]["notify"]["url"]
                .as_str()
                .is_some_and(|u| u.contains(&needle))
        })
        .collect()
}

fn native_token(exec: &ExecMock, session: &str) -> Option<String> {
    native_keep_bodies(exec, session)
        .last()
        .and_then(|b| b["job"]["notify"]["token"].as_str().map(str::to_owned))
}

fn native_scope_id(exec: &ExecMock, session: &str) -> Option<(String, String)> {
    native_keep_bodies(exec, session).last().and_then(|b| {
        Some((
            b["job"]["scope"].as_str()?.to_owned(),
            b["job"]["id"].as_str()?.to_owned(),
        ))
    })
}
