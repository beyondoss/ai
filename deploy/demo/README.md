# The whole agent, on AWS, for anyone to run

One model key and AWS credentials, four commands:

```sh
OPENAI_API_KEY=… ./deploy/demo/up.sh      # brings everything up, prints where the agent is
./deploy/demo/demo.sh                     # a scripted conversation: the checks, a run record, a report
./deploy/demo/demo.sh kill                # SIGKILL the agent mid-job; the session survives
./deploy/demo/down.sh                     # deletes everything, then proves it is gone
```

What runs is the real thing: `agent serve --service` talking to a real model through the
real gateway (`crates/gateway`, with a managed `bai_v1` key, so every call is billed in an `ai.usage`
row), running its tools in a sandbox over the exec protocol, keeping background jobs and learning they
ended through v1.4 job-end notify, with sessions sealed on EFS that outlive the agent.

## Prerequisites

`aws` (configured; the demo uses `us-west-2`, override with `AWS_REGION`), `docker`, `cargo`, `jq`,
`openssl`, `curl`. Docker builds the three images (the gateway and agent from the repo's own
Dockerfiles, the sandbox from `Dockerfile.sandbox`) and pushes them to ECR; the first `up.sh` takes
a while for that, later ones do not rebuild (`REBUILD=1` forces it). The agent's port is opened to your
current public address only: if your address changes, rerun `up.sh`, which adds the new one.

Provider: OpenAI by default (`gpt-5-mini`; `DEMO_MODEL` to change it). `OPENROUTER_API_KEY` instead of
`OPENAI_API_KEY` uses OpenRouter (model `openai/gpt-5-mini`). Anthropic and Bedrock are not used.
`DEMO_NAME` (default `demo`) names the deployment and its tag, so two can coexist.

## The sandbox: Fargate or E2B

By default the sandbox is one Fargate service shared by every session. `SANDBOX=e2b` swaps it for
[E2B](https://e2b.dev) microVMs, **one per session**, which is how a real deployment would run:

```sh
export E2B_API_KEY=…
SANDBOX=e2b OPENAI_API_KEY=… ./deploy/demo/up.sh
SANDBOX=e2b ./deploy/demo/demo.sh all
SANDBOX=e2b ./deploy/demo/down.sh
```

`up.sh` then creates no sandbox service; `demo.sh` creates an E2B sandbox the first time a session
needs one (REST API at `api.e2b.dev`, `X-API-Key` from your `E2B_API_KEY`; the key is passed to curl on
stdin and is never written anywhere). Each VM

- carries the metadata `beyond-ai-demo=<name>` and a 30 minute timeout (`E2B_TIMEOUT`), so `down.sh`
  can find it and it ends by itself if nothing does;
- gets the same static `execd` binary, uploaded through E2B's envd file API and started with
  envd's process API (`EXECD_KEPT_JOBS_SURVIVE_IDLE=1`, so it advertises job-end notify), listening on 8081,
  which E2B exposes at `https://8081-<sandbox id>.e2b.app`;
- is the exec endpoint of its session only: the session's grant carries that URL and a bearer token
  (an exec header, sealed in the grant), and `execd` answers 401 to anything without it, since the
  endpoint is on the public internet. The bearer is uploaded as a file (`EXECD_BEARER_FILE`) that is then root's, mode 0600, and `execd` runs
  as root through the VM's `sudo` with `EXECD_RUN_AS=user`: the commands run as the VM's ordinary user and cannot
  read the bearer file, `execd`'s memory or `/proc/<execd>/environ`; the bearer is on no command line.

Job-end notify across the internet: the agent task announces `https://<its-public-ip-dashed>.sslip.io:8443`
as its `--notify-origin` (computed at every start, so a replacement task announces its new address),
the demo's throwaway CA has a certificate for `*.sslip.io`, and each VM trusts that CA only. Port 8443
is open to the world in this mode; it accepts only a callback token the agent's fleet key minted.
`sslip.io` is a public wildcard DNS service that resolves `1-2-3-4.sslip.io` to `1.2.3.4`.

Two more checks join the list in this mode:
`sandbox-per-session` (each session's commands run in the VM E2B made for it, the ids differ, and a file written in one session's VM is not
visible in another's) and `sandbox-auth-required` (no bearer or a wrong one: 401).

`down.sh` kills every sandbox carrying the demo's metadata and verifies through E2B's list API that none
remain (`e2b_sandboxes` in the teardown table). It does this whenever `E2B_API_KEY` is set, and
refuses to run without it when `SANDBOX=e2b`. E2B's cost is whatever your plan charges for the minutes
the VMs live; a full `demo.sh all` keeps them for about fifteen.

## The three use cases

An agent is a folder, and here each folder is **deployed as a bundle**. `SANDBOX=both
./deploy/demo/up.sh` brings up everything the three folders under `agents/` need (E2B for two of them,
a stand-in managed-MCP endpoint, a private S3 bucket for the bundles, the exec front, and ECS Exec on
the agent); each folder is then run for real, against the real model through the real gateway, with
named checks, a `run.json` and a `report.html` of its own in `results/<timestamp>-<name>/`:

| command             | the folder                                                                                    | checks                                                                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `demo.sh assistant` | `AGENTS.md` (persona, `model`, `tools`, `max-steps`), a `.mcp.json` with DeepWiki and a skill | `persona-applied`, `remote-mcp-real`, `managed-mcp-no-key`, `memory-across-sessions`, `limits-enforced`, `skill-from-bundle`, `no-sandbox-at-all` |
| `demo.sh coding`    | a failing Python app, a skill with a script, a `reviewer` subagent, `AGENTS.md`               | `vm-created-on-first-tool`, `skill-used`, `fix-verified`, `subagent-from-folder`, `app-as-background-job`, `browser-sees-localhost`               |
| `demo.sh browser`   | `AGENTS.md`, a `.mcp.json` and a page                                                         | `browser-reads-page`, `no-held-connection`, `idle-cost`, `state-survives-pause`, `state-survives-idle-reap`                                       |

### A fourth: an agent that acts as a signed-in user

`SANDBOX=oidc ./deploy/demo/up.sh` is a stack of its own (no sandbox, no E2B): the gateway and the agent as
above, and one more task holding **Keycloak** (the identity provider, two users, `alice` and `bob`) and the
**broker** with the notes service it fronts. `demo.sh oidc` signs both users in through the provider's real
login and consent pages (scripted: `oidc/login.sh`), binds each to a session, and drives the agent through
`user-signs-in-once`, `agent-acts-as-user`, `agent-never-sees-token` (including a scan of the agent
process's memory, with a control string planted so the scan is shown to see what is there), `token-refresh`
(the access token lives 60 s), `users-isolated`, `scope-enforced` and `revocation`. The agent folder is
`agents/oidc/`; the broker's behaviour is written down in [BROKER.md](BROKER.md). Use a name of its own
(`DEMO_NAME=oidc`) so it never shares resources with another stack.

### Bundles, and what opening a session costs

`demo.sh` tars the folder (`tar --format=ustar`, with this run's stamps in it), names the tar by its
SHA-256, uploads it to a private bucket (`beyond-ai-demo-<name>-bundles-<account>`, tagged, public access
blocked, emptied and deleted by `down.sh`, which counts it as `s3_buckets`) and mints each grant with
`--bundle <presigned link>=<sha256>`. The replica fetches and verifies the tar and reads `AGENTS.md`, its
frontmatter, `.mcp.json`, skills and subagents from it, so a session no longer asks a sandbox for its own
folder (about 32 round trips at every open without one).

- **assistant**: no sandbox at all. Its grant has an empty `exec_url`; the `read` tool is still there,
  confined to the bundle, which is how it reads its skill. `no-sandbox-at-all` unseals the grants and
  counts the requests every sandbox logged over the whole run.
- **coding**: the grant's `exec_url` is the exec front (`front.py`, a second container in the sandbox
  task). The front is a stable address that costs nothing: on the first request for a session it makes
  that session's E2B microVM, seeds it from the bundle (the VM fetches the tar itself) and forwards that
  request and every later one, streaming. `vm-created-on-first-tool` shows that no VM existed when the
  session opened, and times open to VM create, and VM create to the first tool result. The browser
  needs the VM's address, so the browser check opens a second session on the same VM with a grant that
  names it. `python3 deploy/demo/front_conformance.py` runs the exec front conformance kit against
  `front.py` with a local substrate (no E2B) and lists what the demo front does not do: no tenants or
  VM cap, no agent token, and a VM that dies is noticed by the next request, not at once.
  `python3 deploy/demo/bundle_conformance.py` does the same for the bundle path (`uc_bundle`) with a local
  stand-in for S3: nothing reads the source at build, so unsafe entries become bundles; the S3 link is
  https and signed but lasts 6 h.
- **browser**: the browser connector is the sandbox, and its address is the front too
  (`/mcp/<session>`). The agent offers the browser's tools at open from a list saved in the bundle
  (`.mcp-tools.json`, made at bundle build by `mcp-tools.sh` running `rustwright-mcp` once), so nothing
  dials the front and no VM exists. The model's first browser call reaches the front, which makes the
  VM from an E2B template that is a snapshot of a running browser stack: Chromium (on a debugging port),
  supergateway with its node modules and the `rustwright-mcp` binary are all in it, the bridge is
  already listening, and one throwaway session has warmed their pages. The front seeds the VM from the
  bundle and forwards; the agent then compares the server's real list with the saved one. Nothing is
  fetched or installed at VM start. `rustwright-mcp` attaches to the running Chromium
  (`RUSTWRIGHT_MCP_CDP_ENDPOINT`) instead of launching one, so the first navigate does not pay a
  browser start; one Chromium serves the VM's sessions. Without that, a first call spent 8 s starting
  Chromium and 4 s starting node, both reading a cold disk.
  `vm-created-on-first-tool` shows it, as for coding. `up.sh` builds that template once
  (`e2b_browser_template`: named by a hash of what it installs, the `rustwright-mcp` binary and the
  start command, found and reused if it exists, so a second `up.sh` builds nothing). It belongs to the
  E2B account, not to the run, so `down.sh` keeps it and the sandbox count it proves is still zero;
  delete it from E2B's dashboard if you want it gone. A plain VM (the coding agent's) still installs
  the stack when it first asks for a browser, and then starts Chromium and the bridge the same way.
  The agent reaches the front, a private address, with no switch turned: a `sandbox: true` connector
  may dial the origin of the grant's `exec_url`, and nothing else private.

Every use case's report has a "Session open" section: sandbox requests while each session opened, and
the time from connecting to the first answer.

Before and after, per session, from the committed records (before: `20261020T230210Z-assistant`,
`20261020T231038Z-coding`, `20261020T231305Z-browser`; after: the records beside them from this change):

| use case  | sandbox requests at open, before | after         | open latency, before (derived) | after (measured)                                                                                                                                                                                                                                                                                                                      |
| --------- | -------------------------------- | ------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| assistant | 32                               | 0             | about 0.9 s                    | 302 ms (mean of 3)                                                                                                                                                                                                                                                                                                                    |
| coding    | 38                               | 0             | about 3.0 s                    | 306 ms, with no VM yet; the VM is made 2.7 s later, by the first tool call, and its first result comes 5.4 s after the create began                                                                                                                                                                                                   |
| browser   | 34 (and no others all run)       | 0 (0 all run) | about 3.0 s                    | 311 ms, with no VM yet; the first browser call makes the VM (from a template that snapshots the running browser stack) and returns its result 7.6 s after it started, down from 10.5 s with a template that held only the stack, and 18.2 s with none: VM 3.8 s, bundle 0.6 s, browser stack and bridge 2.0 s (was 5.7 s), call 1.2 s |

Where the browser bridge's start went, timed in a fresh VM (not the front's overlapped path): from a
template that only had the files baked, node and supergateway took 4.1 s to listen, `rustwright-mcp`
50 ms to answer `initialize`, and the first navigate 8.1 s (it launches Chromium); every one of those
is reading a cold disk, and the same steps with the files already read take 1.0 s, 40 ms and 1.7 s.
From the snapshot template the bridge is listening and the first `initialize` plus navigate takes
190-330 ms; what is left of the start is the VM resume (about 1.2 s more than a plain VM, as pages come
in on demand), one round trip to check the stack, and starting the page server for the bundle.

The "before" counts are the exec requests in the old records before the first model call (assistant:
129 across its setup session and three sessions, less the setup's own command, divided by four). The
old runs recorded no open latency, so "before" is the span of that burst of requests (the assistant's
sat on the Fargate host inside the VPC; the coding and browser ones crossed the internet to an E2B VM).
The new runs time the open itself: connecting, then the first answer.

The assistant uses `gpt-4.1-mini` in its `AGENTS.md` while the replica's default is `gpt-5-mini`, so the
billing rows can tell the file's model from the default. The managed connector's real key lives in SSM,
readable by the stand-in's task role only; the checks look for it in the grant (unsealed with the agent's
verifier), the agent's logs and the session transcript, and read the agent process's memory through ECS
Exec as far as that is feasible (the report says what was read). The coding and browser agents run in
private E2B VMs; the browser is `rustwright-mcp` behind supergateway, started by the demo because hosted
mode skips `command` entries in `.mcp.json`, and named in the grant as a connector with `sandbox: true`.
Their checks are in `usecases.sh`, added the same way as any other (see "Adding a check").

## What you will see

`demo.sh` runs a list of named checks and prints each as it goes:

| check                        | what it shows                                                                                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grant-refused`              | a tampered session grant is refused (401) at connect                                                                                                                        |
| `real-model-turn`            | a real model answers through the gateway; first-token latency, tokens                                                                                                       |
| `sandbox-exec`               | the model runs `uname -a; ls /` in the sandbox container, not on the agent                                                                                                  |
| `background-job-notify`      | `sleep 20` runs in the background; the sandbox calls the agent back (`job_ended`) and the agent tells the model: the moment is printed, with the callback-to-notice latency |
| `notify-forged-refused`      | a `job_ended` with the right kid and a wrong tag, sent while the session is open, is refused (401) and changes nothing                                                      |
| `kill-and-recover`           | a 140 s background job starts, the agent task is stopped (ECS SIGKILLs it), ECS replaces it, the same session reattaches from EFS and the job's end is reported             |
| `sandbox-cannot-reach-store` | a TCP connect from inside the sandbox to the EFS mount target's port 2049 fails, and the sandbox mounts no NFS                                                              |
| `sealed-at-rest`             | the phrase you saw does not appear in the session's bytes on EFS, read from a one-off task with the agent's role and group; a hex excerpt is recorded                       |
| `billing-rows`               | every model call has exactly one `ai.usage` row, complete, and the cost adds up                                                                                             |
| `teardown-zero`              | filled in by `down.sh`: every resource type counted 0                                                                                                                       |

`demo.sh` runs the first seven (with `sandbox-cannot-reach-store`) and `billing-rows`; `demo.sh kill` runs `kill-and-recover` and
`billing-rows`; `demo.sh all` runs everything in one record.

Each run writes `results/<UTC timestamp>/run.json` (the commit, region, model, each check with its
timestamps and measurements, a timeline across the gateway, agent and sandbox, the transcript, the
billing rows, the on-disk bytes, and the teardown result) and `report.html`: one self-contained page
(no network, light and dark, readable on a phone) that replays the run: a system map with packets
moving as the events happened, the conversation, the kill, the billing ledger, what is on disk, the
check board and the teardown. `report.sh <dir>` rebuilds a page from its `run.json`. Neither file
ever holds a key or a grant.

### Adding a check

In `demo.sh`: write `check_<name_with_underscores>()` (it talks to the agent with
`chat <session> --prompt …` and ends with `result 1|0 "detail" '{measurements}' '{headline}'`), give it a
title in `CHECK_TITLE`, and add its name to the lists at the top. The runner times and records it and
the report draws it.

## Load: one replica, many tenants

`load.sh` proves that one service-mode replica stays bounded when many tenants share it, on the real
stack and the demo's task size (Fargate, 0.5 vCPU, 1 GB). It spends nothing on a model: `SANDBOX=load`
brings up only the agent task and a second task running `beyond-ai-demo load-host`, a scripted model that
answers instantly (a prompt gets one `read` of the bundle's skill, the tool result gets a one-line answer)
and the bundle host. There is no gateway, no sandbox and no provider key.

```sh
./deploy/demo/load.sh up      # about 4 minutes; DEMO_NAME=load, so it shares nothing with another stack
./deploy/demo/load.sh run     # about 20 minutes: results/<timestamp>-load/{run.json,report.html}
./deploy/demo/load.sh down    # down.sh, with its counts merged into the same record
```

`LOAD_STORE=local` puts the session store on the task's own ephemeral volume instead of EFS and changes nothing else,
so a run with and without it says what the network filesystem costs. The record also carries the replica's
blocking-pool instruments and its per-phase `OPEN` lines (`analysis.waves[].blocking` and `.tail`), and the
raw `driver.json` and `opens.log` sit beside it.

The load: 150 bundles (50 tenants, three versions each, 10 with 5 MB of skills) and 100 more for the
burst. Two waves, each ramping to 500 sessions open at once (60% long-lived; 40% churning through a second
set of bundles, so the cache has something to evict), a few turns each, every turn reading a skill; hold,
close everything, idle 3 minutes. Then the cold-start burst (200 sessions at once on 100 bundles nobody has
opened), then one tenant whose `.mcp.json` lists 200 servers, with the grant limit unset and at 5.
`LOAD_SESSIONS`, `LOAD_IDLE_SECS` and the like change the sizes; `LOAD_CACHE_MB` (default 16) is the cache
budget, set low so eviction happens.

The replica is read from inside its task. A `sampler` container shares the task's process namespace and
network and logs, every two seconds, what the kernel says about the agent process (resident and anonymous
memory, threads, fds, CPU), its bundle cache on disk, the established sockets and the agent's own metrics.
The bundle host holds each fetch for 250 ms, so overlapping fetches show as a peak it can count (S3 shows
none), and counts dials to the connector URLs. The driver records what a client sees. `load-analyze.py`
joins them and grades the bounds; a metric that grows from one wave to the next is a failed check.

Measured on one run (`results/20261024T171544Z-load`, commit in the record):

| bound                                 | result                                                                                                                                                                                                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| memory per session                    | about 270 KB of anonymous memory per session the replica holds (slope over the waves); a closed session is held for the idle reaper's 60 s, so the replica held 1,950 for 500 open at a client, about 1 MB per client-open session                                        |
| open latency, 500 open at once        | p50 1.19 s and p99 5.6 s in wave 1; 1.13 s and 4.3 s in wave 2 (40 ms of it network); the replica's CPU is the limit, 31% median of a vCPU with peaks of 62 to 77% against the task's 50%                                                                                 |
| cold-start burst                      | 200 sessions on 100 unseen bundles: open p50 5.2 s, p99 8.8 s, every session opened                                                                                                                                                                                       |
| concurrent bundle fetches             | peak 4 (the cap), reached in the burst and in both waves                                                                                                                                                                                                                  |
| connector cap, `.mcp.json` of 200     | limit unset: 64 servers dialed per session (the agent's ceiling), 320 connections open at once for 5 sessions; limit 5: 5 per session, 25 for 5 sessions                                                                                                                  |
| no growth across waves                | peak anonymous memory 894 MB then 464 MB; the floor in the idle after each wave 322 MB then 324 MB; threads and fds back to 4 and 12 after both; open latency and CPU the same in both waves                                                                              |
| bundle cache against its 16 MB budget | peak 54 MB with sessions open (bundles a session holds are never evicted); 1,283 directories evicted; **not trimmed when sessions close**: it stayed at 54 MB through both idles and was trimmed only by the next install (19 MB after the burst, still above the budget) |
| memory with no sessions               | **a quiet replica is not quiet**: with zero sessions and no work, anonymous memory drifted up 20 MB in one idle and 58 MB in the other, in step with transparent huge pages (30 and 118 MB of it), and fell to 186 MB only when the replica next did work                 |
| headroom                              | wave 1 peaked at 894 MB of the task's 1,024 MB (wave 2 at 464 MB): about 2,000 held sessions is the ceiling at this size                                                                                                                                                  |

Two things the run found are in the record as failed checks. The cache was trimmed only when a bundle was
installed, so it stayed over its budget with no sessions open; that is fixed in #488 (the record is from before
it). The other is the memory drift with no sessions: the kernel's huge-page collapse filling pages the allocator
has freed but not returned. The agent enables huge pages on purpose, so this is a trade to decide, not a leak,
and it does not grow across waves.

## What it costs

About **$0.08 an hour** while up (three Fargate tasks of 0.25 to 0.5 vCPU, EFS, nothing else: no NAT
gateway, no load balancer), plus the model: a full `demo.sh all` spends about one cent with
`gpt-5-mini`. Tear down when you are done; `down.sh` is quick.

## How the pieces are wired

- **gateway** task: the gateway plus a NATS sidecar (its allowance set is fail-closed until it has read
  one). The only task that references the provider key, from an SSM SecureString; its role can read that
  key and its id-signing key and nothing else.
- **agent** task: `serve --service`, `--notify-origin https://agent.<name>.local:8443`. It is not pid 1
  (a shell is, and ignores SIGTERM), so a stopped task is SIGKILLed with no chance to seal or unlock: a
  hard kill with nothing but `aws ecs stop-task`.
- **sandbox** task: `execd`, the repo's production exec endpoint (`crates/execd`), which passes the
  exec conformance suite, notify included (`crates/execd/tests/conformance.rs`). It listens beyond loopback, so it requires a bearer: `up.sh` stores a random `sandbox-bearer` in SSM, the task gets it as `EXECD_BEARER`, moves it into a root-owned 0600 file and starts `execd` (root, `EXECD_RUN_AS=sandbox`) on that file, so commands run as the unprivileged `sandbox` user and cannot read it, and `demo.sh` seals it into every grant as the exec `Authorization` header. It logs JSON; `execd.jq`
  turns each line into the `exec_request`, `exec_start`, `exec_end`, `job_ended_callback` and
  `listening` events the checks and the report read. It runs agent-controlled code, so it has its own security
  group and no EFS mount: nothing admits it to the session store (see "The session store" below).
- **the session store**: the EFS file system admits one identity, the agent task's role, through the
  agent's access point, over TLS. Three layers say so: the mount target's security group admits NFS (2049)
  from the agent's group alone (the sandbox is in a group of its own, reachable only from the agent's); the
  volume has `iam: ENABLED`, so the mount helper signs in as the task role; and the file system's policy
  allows `ClientMount` + `ClientWrite` only to that role, only with that access point, and denies any
  connection without TLS. A process in the sandbox that spoke NFS by itself (no root needed) would find no
  route, and if it found one, no identity: encryption keeps session contents secret, not intact, so
  anything that could reach the store could delete or roll back any tenant's sessions. `sandbox-cannot-reach-store`
  shows a TCP connect from the sandbox to the mount target failing; `sealed-at-rest` reads the bytes from a
  one-off task with the agent's definition (role and group) instead of from the sandbox.
- **oidc** task (`SANDBOX=oidc`): a Keycloak container (`start-dev`, the realm imported from
  `oidc/realm.template.json` through an SSM parameter) and the broker container (`beyond-ai-demo
  oidc-endpoint`, `crates/demo/src/oidc/`) side by side. The broker's managed port is public like the
  stand-in's; its control port and Keycloak's login page are open to the caller's address alone. Its role
  reads the broker's keys, its client secret and the realm, and nothing else.
- **names**: Cloud Map private DNS (`gateway`, `agent`, `sandbox` in `<name>.local`). A replaced agent
  gets a new address; its name follows after the private zone catches up (some tens of seconds), and the
  sandbox's callback schedule retries until it does. (With E2B the callback names the agent's address
  instead; see above.)
- **callbacks are https**: exec v1.4 only calls back over TLS, so `up.sh` makes a throwaway CA and a
  certificate for `agent.<name>.local`; the sandbox trusts the CA, the agent serves the certificate.
- **keys**: provider key from your environment, and everything generated by `up.sh` (grant signing and
  seal keys, the gateway's signing keys, the TLS key), live only in SSM under
  `/beyond-ai-demo/<name>/`. `demo.sh` reads what it needs from there to mint grants locally
  (`beyond-ai-demo mint-grant`, which uses the repo's own minter and is tested against the agent's
  verifier).

## How teardown is verified

Everything carries the tag `beyond-ai-demo=<name>` and every create looks the thing up first, so
`up.sh` twice changes nothing the second time and an interrupted run is simply rerun. `down.sh` deletes
in dependency order, tolerating what is already gone, then counts what is left: one explicit listing for
each of 21 resource types (SSM parameters by prefix, IAM roles by name, log groups, ECS clusters,
services, tasks and task definitions in every status, Cloud Map namespaces and services, the Route 53
zone, EFS file systems, mount targets and access points, the VPC, subnets, gateways, route tables,
security groups and network interfaces, ECR repositories, the bundles bucket) and the tagging API. It exits non-zero naming
anything that remains. ECS keeps deleted clusters, services and stopped tasks in the tagging index for
hours, so a tagging-API entry only counts if the service's own API cannot show it gone; the count that
was only index lag is reported beside it.
