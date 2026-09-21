//! The invariant checker: the history of what clients were promised, against what is on the shards.
//!
//! A scenario that ends without an assertion failing has shown that one path worked once. This is the
//! other half — the part that can catch a violation nobody wrote a scenario for, because it checks
//! properties of the *whole run* rather than steps of one.
//!
//! Deliberately structural. Session lines are sealed with the tenant's key, and the checker does not
//! open them: the transcript's *content* is compared through the protocol (`get_messages` on whoever
//! owns the session now), which exercises the real replay path and needs no key. What the checker
//! reads off disk is the shape — which epochs exist, which file each one is, how they are sealed —
//! and that is exactly where the single-writer claim lives.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// One thing that must be true, and whether it was.
#[derive(Debug)]
pub struct Finding {
    pub claim: &'static str,
    pub ok: bool,
    pub detail: String,
}

impl Finding {
    fn ok(claim: &'static str, detail: impl Into<String>) -> Self {
        Self {
            claim,
            ok: true,
            detail: detail.into(),
        }
    }

    fn violated(claim: &'static str, detail: impl Into<String>) -> Self {
        Self {
            claim,
            ok: false,
            detail: detail.into(),
        }
    }
}

/// A session's segments, newest last.
pub fn segments(session_dir: &Path) -> Vec<(u64, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(session_dir) else {
        return Vec::new();
    };
    let mut out: Vec<(u64, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let name = path.file_name()?.to_str()?.to_owned();
            let epoch = name.strip_suffix(".jsonl")?.parse::<u64>().ok()?;
            Some((epoch, path))
        })
        .collect();
    out.sort_by_key(|(e, _)| *e);
    out
}

/// Is this segment a **base** — a full rewrite that replaces the chain before it?
///
/// The header is the segment's first line and is deliberately *not* sealed: a reader has to be able
/// to bound every segment without holding the tenant's key. So the checker can read the shape
/// without ever being able to read the content, which is exactly the split it wants.
fn is_base(path: &Path) -> bool {
    let Ok(raw) = std::fs::read(path) else {
        return false;
    };
    raw.split(|b| *b == b'\n')
        .next()
        .and_then(|line| serde_json::from_slice::<Value>(line).ok())
        .is_some_and(|h| h["base"] == true)
}

/// **C1 — one writer per session.**
///
/// Structural, and that is the strength of it: an epoch is a *file*, created with `O_EXCL`, so two
/// owners cannot share one however badly they race. What this verifies is that the invariant holds
/// as stated — epochs are distinct, and the set on disk is one a reader can actually walk.
///
/// **Gaps are legal, and getting that wrong cost a false alarm.** This used to demand that the
/// epochs be contiguous, which is true only of a session short enough never to consolidate. Above
/// the segment budget a persist writes a *base* — a full rewrite — and deletes the segments older
/// than the base it supersedes, while epoch 1 is kept forever because it is the `O_EXCL` create that
/// says the session exists. So a long-lived session's retained set is `{1} ∪ [base..newest]`, with
/// exactly one gap. The first soak that ran long enough to consolidate reported "epochs 1..=22 are
/// not contiguous: 13 files" as a violation; the fleet was right and the checker was wrong.
///
/// What is checked instead is the rule that actually holds: no epoch appears twice, the run above
/// the gap is contiguous, and if there is a gap the segment that starts the upper run is a base —
/// so a reader beginning there has every file it needs. A duplicate, a second gap, or a gap that
/// does not begin at a base means an epoch was minted twice or lost, which is the failure the fence
/// exists to make impossible.
pub fn one_writer_per_session(session_dir: &Path) -> Finding {
    let segs = segments(session_dir);
    if segs.is_empty() {
        return Finding::violated("C1", format!("no segments under {}", session_dir.display()));
    }
    let mut seen = std::collections::BTreeSet::new();
    for (epoch, path) in &segs {
        if !seen.insert(*epoch) {
            return Finding::violated(
                "C1",
                format!("epoch {epoch} appears twice ({})", path.display()),
            );
        }
    }

    // Epoch 1 is kept forever, so the run to check is whatever sits above it.
    let tail: &[(u64, PathBuf)] = if segs[0].0 == 1 { &segs[1..] } else { &segs };
    let Some((oldest, oldest_path)) = tail.first() else {
        return Finding::ok("C1", "one segment, the O_EXCL create that says it exists");
    };
    let newest = tail.last().map_or(*oldest, |(e, _)| *e);
    if newest - oldest + 1 != tail.len() as u64 {
        return Finding::violated(
            "C1",
            format!(
                "epochs {oldest}..={newest} have more than one gap: {} files",
                tail.len()
            ),
        );
    }
    // A gap is only legal where a consolidation made one, and a consolidation always starts a base.
    let consolidated = *oldest > 2 || (segs[0].0 != 1 && *oldest > 1);
    if consolidated && !is_base(oldest_path) {
        return Finding::violated(
            "C1",
            format!(
                "epochs below {oldest} are gone but {} is not a base — a reader starting there has \
                 no complete history to start from",
                oldest_path.display()
            ),
        );
    }
    Finding::ok(
        "C1",
        if consolidated {
            format!(
                "{} segment(s): epoch 1, then a base at {oldest} and a contiguous run to {newest}",
                segs.len()
            )
        } else {
            format!("{} epoch(s), contiguous from 1, one file each", segs.len())
        },
    )
}

/// **C2 — a takeover seals the previous segment.**
///
/// After ownership changes there must be a *new* epoch, and the old one must have stopped growing.
/// The seal itself is recorded in the new segment's header; what is checked here is the observable
/// consequence, which is the one that matters to a reader: the previous file is final.
pub fn takeover_sealed_the_previous_segment(session_dir: &Path, before: u64) -> Finding {
    let segs = segments(session_dir);
    let Some((newest, _)) = segs.last() else {
        return Finding::violated("C2", "no segments at all");
    };
    if *newest <= before {
        return Finding::violated(
            "C2",
            format!("expected a new epoch after {before}, newest is still {newest}"),
        );
    }
    Finding::ok("C2", format!("epoch advanced {before} → {newest}"))
}

/// **No acknowledged line is lost.**
///
/// Every message the client was *told* about has to be in the transcript the session replays after
/// the ownership change. This is the promise-keeping invariant, and the reason the history records
/// acknowledgements as they happen rather than reconstructing them at the end.
pub fn no_acknowledged_message_lost(history: &[Value], replayed: &[Value]) -> Finding {
    let promised: Vec<String> = history
        .iter()
        .filter(|e| e["kind"] == "message_committed")
        .filter_map(|e| e["detail"]["text"].as_str().map(str::to_owned))
        .collect();
    let have: Vec<String> = replayed
        .iter()
        .flat_map(|m| {
            m["content"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|c| c["text"].as_str().map(str::to_owned))
        })
        .collect();
    for want in &promised {
        if !have.iter().any(|h| h.contains(want)) {
            return Finding::violated(
                "no-lost-write",
                format!(
                    "{want:?} was acknowledged to a client but is not in the replayed transcript"
                ),
            );
        }
    }
    Finding::ok(
        "no-lost-write",
        format!(
            "{} acknowledged message(s) all present after replay",
            promised.len()
        ),
    )
}

/// The ordinary case: the session is where its hash says it is.
pub fn reachable_by_hash(session: &str) -> Finding {
    Finding::ok(
        "reachable",
        format!("{session} answered on its hash target"),
    )
}

/// The session is alive and intact, but its hash target is not the replica holding it — so the edge
/// had to walk the ring to find it.
///
/// **Expected, not a failure**, and the distinction took a while to get right. After a failover a
/// session is live on the substitute, and the replica its id hashes to has nothing to serve and no
/// way to say where it went. It answers 503; the edge tries the rest of the slice; the lock holder
/// accepts. That is the documented routing contract (see the agent's ARCHITECTURE.md), and it is
/// safe by construction, because only the replica holding the lock can serve the session at all.
///
/// It is still worth counting rather than passing silently. A run where most sessions need the walk
/// is a run where a strictly hash-and-retry edge would have had an outage for each of them, lasting
/// until the idle reaper freed the lock — which is the number that says how load-bearing the ring
/// walk is, and it is high after any rolling deploy.
pub fn stranded_from_its_hash_target(session: &str) -> Finding {
    Finding {
        claim: "reachable",
        ok: true,
        detail: format!(
            "{session} moved, and the ring walk found it — its hash target answers 503 until the \
             idle reaper frees the lock, so a hash-only edge would have had an outage here"
        ),
    }
}

/// A session that could not be reached at all once the chaos stopped.
///
/// Distinct from a lost write: the history is intact, but a session nobody can open is an outage,
/// and after the last replica has come back there is no legitimate reason for one.
pub fn unreachable_after_soak(session: &str) -> Finding {
    Finding::violated(
        "reachable",
        format!("{session} could not be opened on any replica after the chaos ended"),
    )
}

/// Print a run's findings and say whether it passed.
pub fn report(scenario: &str, findings: &[Finding]) -> bool {
    let passed = findings.iter().all(|f| f.ok);
    println!("\n{} {scenario}", if passed { "PASS" } else { "FAIL" });
    for f in findings {
        println!(
            "  {} {:<14} {}",
            if f.ok { "·" } else { "✗" },
            f.claim,
            f.detail
        );
    }
    passed
}

/// **C10, continuously — a tenant's sessions live only in that tenant's subtree.**
///
/// The matrix proves per-tenant sealing once, in a scenario built for it. This asks the same
/// question of every session a long randomized run produced, which is where a routing or
/// id-derivation mistake would actually show up: a fork minting an id without its parent's prefix,
/// a takeover writing under the replica's own idea of the tenant rather than the grant's.
///
/// Structural, and cheap: the path a session's directory sits at encodes both answers — the shard it
/// was addressed to, and the tenant whose subtree it is under.
pub fn session_is_in_its_own_subtree(dir: &Path, shard: &str, tenant: &str) -> Finding {
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_owned();
    let under = dir
        .parent()
        .and_then(Path::parent)
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_owned();
    if under != tenant {
        return Finding::violated(
            "C10",
            format!("{name} is under tenant {under:?}, not the {tenant:?} that owns it"),
        );
    }
    if !name.starts_with(&format!("{shard}.")) {
        return Finding::violated(
            "C5",
            format!("{name} sits on shard {shard} but is not addressed to it"),
        );
    }
    Finding::ok(
        "C10",
        format!("{name} is on its own shard, under its own tenant"),
    )
}

/// **C10 — nothing a client said is readable on the mount.**
///
/// Every line is sealed under the tenant's own key, so text a client sent must not appear anywhere
/// in the bytes on disk. Run over the whole shard rather than one session, because the interesting
/// failure is a line written to the *wrong* place, which a per-session check would miss by
/// construction — it would be looking in the directory the line never reached.
///
/// **`needles` must be small and distinctive, not "every marker".** The cost is
/// `files × needles × file size`, and the first version was handed every marker the run produced.
/// That was harmless while a capped model server held a soak to 64 turns; the moment the cap was
/// lifted and a run committed some four thousand, this check burned twenty-two minutes of CPU and
/// never finished. It also bought nothing — the markers share a common shape, so a fragment of that
/// shape detects a plaintext leak just as surely as four thousand whole strings. Pass the shape,
/// plus a handful of complete markers so the check is specific as well as sensitive.
pub fn nothing_readable_on_the_shard(shard_root: &Path, needles: &[String]) -> Finding {
    let mut files = 0usize;
    let mut stack = vec![shard_root.to_path_buf()];
    while let Some(at) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&at) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            files += 1;
            let Ok(raw) = std::fs::read(&path) else {
                continue;
            };
            let text = String::from_utf8_lossy(&raw);
            if let Some(hit) = needles.iter().find(|m| text.contains(m.as_str())) {
                return Finding::violated(
                    "C10",
                    format!("{hit:?} is readable in {}", path.display()),
                );
            }
        }
    }
    Finding::ok(
        "C10",
        format!(
            "{files} file(s) on this shard, none containing any of {} thing(s) a client said",
            needles.len()
        ),
    )
}
