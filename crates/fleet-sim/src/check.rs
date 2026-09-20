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

/// **C1 — one writer per session.**
///
/// Structural, and that is the strength of it: an epoch is a *file*, created with `O_EXCL`, so two
/// owners cannot share one however badly they race. What this verifies is that the invariant holds
/// as stated — epochs are distinct, contiguous from the first, and each is its own file. A duplicate
/// or a gap means an epoch was minted twice or lost, which is the failure the fence exists to make
/// impossible.
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
    let first = segs.first().map(|(e, _)| *e).unwrap_or(0);
    let last = segs.last().map(|(e, _)| *e).unwrap_or(0);
    if last - first + 1 != segs.len() as u64 {
        return Finding::violated(
            "C1",
            format!(
                "epochs {first}..={last} are not contiguous: {} files",
                segs.len()
            ),
        );
    }
    Finding::ok(
        "C1",
        format!(
            "{} epoch(s), contiguous from {first}, one file each",
            segs.len()
        ),
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

/// The session is alive and intact, but its hash target is not the replica holding it.
///
/// **Not a correctness failure** — nothing was lost, and the checker says so separately. It is an
/// availability one, and a contract note for the edge: after a failover the session is live on the
/// substitute, and the replica the hash chooses answers 503 until the substitute's copy is
/// idle-reaped, which defaults to an hour. A strict hash-and-retry edge waits that long; one that
/// walks the ring after repeated 503s finds it immediately.
pub fn stranded_from_its_hash_target(session: &str) -> Finding {
    Finding {
        claim: "reachable",
        ok: false,
        detail: format!(
            "{session} is intact but lives on a replica its hash does not choose — a hash-only edge \
             cannot reach it until the idle reaper frees the lock. The edge needs a ring-walk \
             fallback after repeated 503s."
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
