# Running the agent as a fleet

`serve --service` turns one agent process into a replica of a multi-tenant fleet. This document is
for the two people who need more than [ARCHITECTURE.md](ARCHITECTURE.md)'s mechanism: whoever
**builds the edge** that routes to these replicas, and whoever **operates** them.

- [What it is for](#what-it-is-for)
- [The edge contract](#the-edge-contract) — obligations this repo does not implement and cannot enforce
- [Operating a fleet](#operating-a-fleet) — sizing, timers, deploys, what to alert on
- [How it scales](#how-it-scales)
- [What it costs, and against what](#what-it-costs-and-against-what)

Everything with a number in it was measured. Where a number is derived rather than observed, it says
so — a capacity figure nobody can trace is worse than no figure, because it gets planned against.

## What it is for

Today's agent runs **inside** the sandbox: one VM per session holds the agent loop, its tools, the
workspace and the transcript. That is a good shape for one user and a bad shape for a service. The VM
must be up for the session to be reachable, its local disk is the only copy of the transcript, and
the cost floor per idle session is a VM.

Service mode moves the loop **out**. The sandbox keeps the workspace and a small exec shim; the agent
becomes a fleet, any replica of which can serve any session; the transcript moves to shared storage.
What that buys, in order of why it was done:

1. **A session survives losing the replica serving it.** Only an in-flight run is lost, back to its
   last committed turn. Nothing else in the design costs as much as this one property.
2. **Idle sessions stop costing a VM.** A detached session is a directory on shared storage and,
   briefly, a session lock. Live-but-idle costs roughly 0.7 MB of replica memory (measured).
3. **Deploys stop being outages.** A drain finishes work in flight while refusing new placements.
4. **One tenant's blast radius is one tenant.** Every line on disk is sealed with that tenant's key,
   which arrives only inside a grant and is never written down.

What it deliberately does **not** do: the agent has no part in routing, service discovery, sandbox
lifecycle, tenant allocation or billing. It knows sessions, namespaces, shard-name→path mappings, a
workspace root, an exec URL and keys. If a change needs a word outside that list, it belongs in the
control plane. See ARCHITECTURE.md for the full contract ledger.

## The edge contract

**The agent does not route.** It answers for the sessions it owns and refuses everything else, and it
never forwards. That makes the edge responsible for properties the agent cannot enforce and this repo
cannot test for you — so they are written here as requirements, with the failure each one prevents.

`crates/fleet-sim/src/edge.rs` is a working reference implementation of all of this, used to grade
every claim in `deploy/efs-proof/results/`. If your edge behaves like that one, the fleet behaves the
way the measurements say it does.

### What a replica answers

| Status                   | Meaning                                                                                                                                                       | What the edge must do                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `101`                    | Upgraded; this replica owns the session                                                                                                                       | Nothing — proxy it                                                                               |
| `503` + `Retry-After: 1` | **Retryable.** The lock is held elsewhere, this replica is at `--max-live-sessions`, the session's previous task is still exiting, or the replica is draining | Retry, per the algorithm below                                                                   |
| `421`                    | This replica does not mount that session's shard                                                                                                              | **Do not retry here.** Re-place onto the slice that does — the shard prefix in the id says which |
| `401` / `403`            | The grant did not verify, or named a tenant or session it may not have                                                                                        | Fail the client. Never retry, never try another replica                                          |
| `400`                    | Malformed request or grant                                                                                                                                    | Fail the client                                                                                  |

A `503` body carries a human-readable reason (`"that session is open on another replica; retry"`,
`"this replica is at its live-session limit; retry"`). Log it; do not parse it.

### Requirement 1 — consistent hashing, never `hash % len`

The edge picks a replica within a slice by hashing the session id onto a ring with virtual nodes (the reference
edge uses 64 per replica).

**Not modulo.** With `hash % len` the ring is renumbered whenever the replica count changes, so
_every_ session moves when one replica joins or leaves. A session that moves while its previous owner
still holds the lock is answered `503` by its new target, and the retry — which is deterministic, and
so returns to that same target — waits for a lock that will not free until the old owner's session is
idle-reaped. The soak found exactly this: sessions unreachable after a replica came back, because the
arithmetic had reshuffled them.

### Requirement 2 — retry the same target first, then walk the ring

A `503` is not "try someone else". It is "the session belongs here and is not ready yet", which is
the common case during a deploy: the previous owner is still exiting and the lock frees in
milliseconds.

But retrying the same target **forever** is also wrong. If the session has already failed over, it is
live on a _substitute_, and its hash target has nothing to serve and no way to say where it went —
the agent never forwards and never hints, by design. So the retry must escalate:

1. Retry the hashed target while it answers `503`.
2. After repeated refusals, **walk the ring** from the hashed position — owner first, then the next
   candidate, and so on across the slice.

This is safe because only the lock holder can serve the session at all: a walk cannot produce two
writers, it can only find the one that exists. Keeping the walk _ordered from the hash_ is what makes
a session sticky to one replica whenever that replica is up, so the behaviour converges instead of
oscillating.

**Measured:** after chaos on the homelab, **7 of 9** sessions needed the walk. On real ECS across 13
rolling deploys and 7 hard kills, **5 of 30** did.

### Requirement 3 — a retry budget longer than the storage lease

An owner that dies **ungracefully** — SIGKILL, host loss, or cut off from storage — cannot release
its session lock. The lock survives until the filesystem lease lapses, and only then can a survivor
take over. That is an availability cost, not a correctness one: the fence is in the data, so history
is safe either way (see ARCHITECTURE.md, "Ownership: the epoch fence").

The edge's retry budget therefore has to exceed that window, or it will give up on sessions that were
about to become available. The reference edge uses **120 s** (`RETRY_BUDGET`).

**Measured on real EFS:** a survivor took over **73.1 s** after a hard kill, and **88.4 s** after the
owner was partitioned from storage — with 44 individual `503`s along the way. On the homelab against
a stock 90 s Linux lease: 77.3 s and 102.5 s. A graceful stop costs neither: the lock is released
cleanly and takeover is **~10 ms** (homelab; on EFS that shows up as 13 deploys with zero
refusals rather than a directly isolated latency).

So: **a budget under ~90 s will strand sessions that a longer one would have served.** Note also that
"0 placements refused" is not "no one waited" — it means nobody gave up.

### Requirement 4 — route the slice from the id prefix

A session id is `<shard>.<opaque>`, and shard names are slice-qualified (`a07-s3`). One load-balancer
rule per slice prefix picks the replica group; the agent answers `421` if it is handed a shard it
does not mount and **never** forwards across slices. Placement is written into the id at mint time
and never changes, so there is no lookup table and nothing is ever rebalanced.

### Requirement 5 — probes

`/livez` and `/readyz` share the tenant-facing listener and need no grant.

- **`/readyz` → the readiness probe.** 503 while a shard is missing, read-only, or hanging; 503 while
  draining. This is what takes a replica out of rotation.
- **`/livez` → the liveness probe.** 200 whenever the process is serving, _including while draining
  and while a mount hangs_.

**Never wire them the other way round.** Restarting a replica does not put a missing mount back, and
killing one mid-drain is the outage the drain exists to avoid.

## Operating a fleet

### The three timers, and which one a long turn threatens

| Timer                                  | Default                                                      | Renewed by                                                            | Does a long-running turn threaten it?                                                                                                |
| -------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Storage lease (holds the session lock) | ~90 s stock Linux; EFS's is undocumented and shorter         | The **kernel's** NFSv4 state-manager thread, per _client_, on a timer | **No.** Renewal is independent of application activity. The agent heartbeats nothing; an flock is held by an open fd                 |
| Idle reaper (`--session-idle-timeout`) | **60 s** in service mode, **1 h** for the single-user daemon | —                                                                     | **No.** `is_reapable` requires _detached_ for the full window **and** `!running`; the clock starts when the last connection detaches |
| Drain grace (`--drain-grace`)          | `0` (immediate); the fleet runs 30 s                         | —                                                                     | **Yes — this is the one.** A turn that outlives the grace is SIGKILLed, and its sessions fall into the ungraceful takeover path      |

So a five-minute model response is safe from the lease and safe from the reaper, and is precisely
what can turn a free rollout into a paid one. **Set `--drain-grace` above your longest expected
turn, and the orchestrator's kill timeout above that** (`terminationGracePeriodSeconds`, or ECS
`stopTimeout` — which is capped at **120 s**, so ~110 s of grace is the ceiling on Fargate).

One consequence worth knowing: the epoch fence is re-checked **once per committed batch — per turn —
not on a timer**. An owner fenced in the middle of a long turn does not discover it until it commits.
That is by design and bounded by one round trip, but "how long until a fenced owner notices" scales
with turn length.

### Sizing a replica

Start from what a running session actually costs a replica. A turn takes **~37 ms of replica CPU**
(derived: 26.9 turns/s per vCPU, measured on 1-vCPU Fargate tasks with a mock model answering
instantly) against **seconds** of model streaming. A running session is idle from the replica's point
of view something like 99% of the time, and that ratio is the whole reason this shape works.

|                                                | Guidance                                                                          | Bound by                                                                                             |
| ---------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Running** (turn in flight)                   | **~270 per vCPU**, assuming a turn every ~10 s                                    | CPU; memory lands in the same place (270 × 4 MiB replay cap ≈ 1.1 GB)                                |
| **Live** (attached or recently detached, idle) | **~3,700 per 4 GB**; the `--max-live-sessions` default of **20,000** wants ~16 GB | Memory at 4 GB; descriptors at the default (2 per session, inside the filesystem's per-client limit) |
| **Stored**                                     | No practical ceiling — ~933 bytes/turn, so a 200-turn session is ~190 KB          | Not capacity. See the caveat below                                                                   |

**Caveats, because these will get planned against.** The 270 figure assumes those Fargate tasks were
CPU-saturated, which _was not measured_ — no CPU metric on the tasks. The soak also absorbed 13
deploys and 7 kills, which depressed throughput, so 26.9 turns/s per vCPU is a lower bound and 270 is
conservative. The 3,700 figure extrapolates a memory slope measured across a 15-session delta
(361 → 372 MB) by more than two orders of magnitude: trust the order, not the digits.

**The unquantified one:** sessions live at `<shard>/<tenant>/sessions/<id>/`, and `list_sessions`
scans that directory. Nobody has measured `readdir` on a tenant with 100k sessions on a network
filesystem. That is the real ceiling on _stored_ sessions per tenant, and it is why the session
catalog is flagged in ARCHITECTURE.md's contract ledger as possibly belonging to the control plane
instead.

### The number that should actually decide your replica size

Not any of the above — **blast radius.** A replica holding N live sessions that dies takes all N into
a 73–88 s takeover. At the 20,000 default that is 20,000 sessions unreachable for a minute and a
half. This argues for **many small replicas over few large ones**, and it is the reason to set
`--max-live-sessions` far below what the descriptors allow.

It applies to _unplanned_ loss only. A rolling deploy costs ~0: the draining replica keeps serving
what it owns and only new placements move. Thirteen consecutive deploys during the EFS soak produced
**0 placements refused**.

### What to alert on

The scrape is loopback-only by design — it describes every tenant on the replica, and the replica is
reachable by tenants, so a routable bind is _refused_ rather than warned about. Run the scraper
beside it (in an `awsvpc` task, a sidecar shares the network namespace).

| Signal                                             | Why                                                                                                                                                                                               |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rate(agent_refusals_total{reason="unavailable"})` | Sustained non-zero means sessions are waiting on locks — a failover in progress, or an edge whose retry budget or ring walk is wrong                                                              |
| `agent_refusals_total{reason="misdirected"}` > 0   | The edge is routing a shard to a slice that does not mount it. Should be flat zero                                                                                                                |
| `agent_refusals_total{reason="auth"}`              | Grant minting or key rotation is broken                                                                                                                                                           |
| `histogram_quantile(.99, agent_lock_wait_seconds)` | Where a dead owner's lease shows up. A rising tail is failover latency, not an error                                                                                                              |
| `agent_sessions_live` vs `--max-live-sessions`     | Capacity headroom, and your blast radius                                                                                                                                                          |
| `agent_runs_in_flight`                             | **The one that predicts memory** — an idle live session costs a few hundred KB, a running one additionally holds up to 4 MiB of replay buffer                                                     |
| `agent_threads` flat                               | Sampled per scrape. A climb while a mount is hanging means the readiness probe is leaking blocking-pool threads; flat is the claim (measured: **0** leaked across 20 probes against a dead mount) |
| `agent_ready_probe_seconds`                        | A mount degrading before it stops answering                                                                                                                                                       |
| `agent_sessions_superseded_total`                  | An owner discovering it was fenced. Should be rare; a rise means replicas are being lost or partitioned                                                                                           |

No tenant, session, shard or workspace identifier appears in any label — asserted by a unit test and
re-checked at runtime against a live replica that has actually served a session.

### Deploying

1. Roll one replica at a time, or a small batch. Drain is per-replica and sessions re-place onto the
   rest of the slice.
2. `--drain-grace` must be **under** the orchestrator's kill timeout and **over** your longest turn.
3. Watch `agent_sessions_live` on the draining replica fall to zero rather than a clock.
4. Adding a shard to a slice needs a rolling restart of that slice's replicas so they mount it.
   Opening a new slice does not touch the existing ones at all.

## How it scales

The unit of scale is a **slice**: at most 10 shards, plus the replicas that mount exactly those
shards. Ten because ECS allows 10 EFS volumes per task. Slices share nothing — no cross-slice traffic
and no shared state — and each tenant is pinned to one. The agent never learns the word.

```
prefix a07-*  ─→  slice a07: replicas × N (autoscaled)  ─→  shards a07-s1 … a07-s10
prefix a08-*  ─→  slice a08: its own replicas           ─→  its own shards
```

**Grow a slice** (under 10 shards): add a file system, roll the slice's replicas to mount it, then
start allocating new sessions to it. **Open a slice** (at 10): a new replica service from the same
image, one more load-balancer rule, and pin new tenants to it. Existing tenants stay where they are.
Nothing moves and nothing is rebalanced, because placement is in the id.

### Where the ceiling actually is

Storage is **not** the constraint, and it is worth being concrete about how far from it we are. Three
saturated replicas generated **~81 appends/s** — under 1% of a _single_ file system's default write
quota — and that was with a mock model answering instantly, which is roughly 27× more write-intensive
per session than production.

| Step                                                | Concurrently running              | Bound by                                                  |
| --------------------------------------------------- | --------------------------------- | --------------------------------------------------------- |
| One replica, 1 vCPU / 4 GB                          | ~270                              | Replica CPU                                               |
| One file system, default quota (50k write IOPS)     | ~100k                             | Write IOPS — and reaching it needs ~370 vCPUs of replicas |
| One file system, quota raised 10× (via AWS support) | ~1M                               | ~3,700 vCPUs of replicas                                  |
| One slice (≤10 file systems)                        | ~1M at default quota, ~10M raised | Nothing in the design — you run out of replicas first     |
| One region                                          | ~100 slices                       | Soft limit on file systems                                |

Derived from 50k write IOPS per file system, ~5 NFS operations per append, and one append per turn
at a turn every ~10 s per running session. The replica row is the measured 270/vCPU.

In practice the fleet is bounded by **how many replicas you are willing to run**, and after that by
slices, which are linear and share nothing. Throughput per session on EFS measured at **2.69 turns/s
against 3.18 on local NFS — 85%**; the raw fleet rates are not comparable because the two runs were
different shapes.

## What it costs, and against what

At roughly 1M sessions/month, EFS came out at **~$1.5k/month all-in** against **$6–22k** for a
ClickHouse-backed design. Two things decided it, neither of them the headline price:

- **Per-tenant encryption kills a column store's advantages.** Every line is sealed with the tenant's
  own key, so ClickHouse cannot compress across tenants and cannot index or query content. You pay
  for a database and get a blob store.
- **The session store already assumes POSIX.** Append, `fsync`, `rename`, `O_EXCL`, advisory locks.
  Keeping that is why there is no storage abstraction in the codebase — the single writer is fenced
  _through the data_ rather than through a lock service.

**S3 Files was rejected** on limits rather than price: locks cap at 8,192 per mount across 256
file-process pairs, and writes bill at a 32 KiB minimum against our measured **933 bytes per turn** —
a 35× write amplification on the dominant operation.

### The EFS settings that decide whether any of this holds

- **Elastic throughput, not the Bursting default.** The soak sustains ~178 KB/s. Bursting's baseline
  scales with _stored_ size, and a fleet's filesystem is small relative to its write rate, so it
  would spend burst credits and then throttle — and every throughput number would become a reading of
  EFS's credit balance rather than of the design.
- **One AZ.** Mount targets are per-AZ; cross-AZ is $0.01/GB each way _and_ adds latency to the
  replay path, which is exactly the path a takeover walks.

A fleet of 3 replicas (1 vCPU / 4 GB) plus a driver, as run for the proof, costs about **$0.33/hr**.
EFS storage and ECR are rounding errors at that size.

## Where the evidence is

- `deploy/efs-proof/` — the proof harness, `up.sh` / `down.sh` / `fault.sh`, and `results/` holding
  the full output of the run these numbers come from: 9 scenarios, 0 skipped, plus a 145,244-turn
  soak through 13 rolling deploys and 7 hard kills.
- `deploy/attached-local/` — the same nine scenarios against a local NFS fleet, for free, on one box.
- `crates/fleet-sim/` — the simulator and the reference edge. `fleet-sim list` names every scenario
  and the claim it grades.
