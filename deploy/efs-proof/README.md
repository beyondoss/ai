# The EFS proof

The fleet simulator's nine scenarios, run once against **real ECS tasks on real EFS** rather than
against the loopback NFS substrate on the homelab. Every claim the storage design makes was already
green locally; this answers the three questions a kernel NFS server we control cannot.

```
./up.sh --images     # build and push the images, then provision and launch the fleet
./up.sh              # provision and launch (idempotent: everything is found by tag first)
./down.sh            # give it all back, and verify that it did
```

`up.sh` prints the exact `fleet-sim matrix` command to run **from inside the driver task**, which is
where it has to run: the checker reads the shards through the local filesystem, and the driver hosts
the model server and exec double that the replicas dial back into.

## What it answered

|                                          | Homelab            | EFS            |
| ---------------------------------------- | ------------------ | -------------- |
| Full session takeover after a partition  | 102 s              | **88.4 s**     |
| Takeover after a hard kill               | 81–89 s            | **73.1 s**     |
| `/readyz` while the mount hangs          | bounded at 1.003 s | **1.0028 s**   |
| Probe threads leaked over 20 such probes | 0                  | **0**          |
| ESTALE during consolidation              | 0                  | **0**          |
| Turns/s per session                      | 3.18               | **2.69 (85%)** |

`9 passed · 0 failed · 0 skipped`, and a half-hour soak of 30 sessions over 5 tenants committing
**145,244 turns** through 13 rolling deploys and 7 hard kills with **0 placements refused** and no
acknowledged write lost. Full output in `results/`.

**The headline is that EFS recovers a lease faster than a stock Linux NFS server**, not slower. The
102 s partitioned takeover was the number gating the Fargate-vs-EC2 decision, and it is not a reason
to avoid Fargate; the edge's 120 s retry budget covers both with room.

## The three things that make this work at all

**A subnet per replica, and the mount target in the driver's.** A partition is a NACL deny rule, and
a NACL is only evaluated at a subnet boundary — so a replica sharing a subnet with the mount target
could not be cut off from it. NACLs rather than security groups because NACLs are **stateless**: an
already-established NFS connection is cut on the next packet. A security group is stateful, and
whether revoking a rule tears down a tracked flow is exactly the sort of thing that produces "the
design strands sessions" results that are really rig artifacts. Verified before anything depended on
it: `/readyz` went 200 → 503 within 10 s on the cut replica while its peer stayed 200 for the full
minute, and the cut replica rejoined by itself 40–50 s after the rule was removed.

**The replica task definition overrides the image's entrypoint.** `Dockerfile.agent` execs the
agent, so the agent is PID 1 — correct for production, and fatal for the `kill` fault, because the
kernel refuses SIGKILL to a PID namespace's init from inside it. The override backgrounds the agent
under a shell that forwards SIGTERM. The shipped image is untouched.

**The driver image is Debian.** `aws ecs execute-command` needs the Session Manager plugin, which is
how a SIGKILL gets inside a Fargate task at all — ECS has no "signal this container" API and
StopTask always sends SIGTERM first. AWS publishes that plugin for glibc only.

## Two EFS settings that decide whether the numbers mean anything

**Elastic throughput, not the Bursting default.** The soak writes ~933 bytes per turn, so it
sustains ~178 KB/s. Bursting's baseline scales with _stored_ size and this filesystem holds ~0.2 GB:
it would spend burst credits and then throttle, and the throughput number would be a reading of
EFS's credit balance rather than of the design.

**One AZ.** Mount targets are per-AZ; cross-AZ costs $0.01/GB each way _and_ adds latency to the
replay path, which is precisely what the takeover measurement is trying to isolate.

## Cost, and why `down.sh` is a gate

About **$0.33/hr** running: 3 replicas at 1 vCPU/4 GB, a driver at 2 vCPU/8 GB, and an internet
gateway. EFS storage and ECR are rounding errors. The experiment is cheap; **the risk is leaving it
up**, at ~$250/month — a hundred times the cost of the run. So `down.sh` ends by listing what still
carries the `fleetsim=efs-proof` tag and exits non-zero if anything does. It earned that on its
first run, which left three resources behind because a custom NACL cannot be deleted while its
subnet still exists.
