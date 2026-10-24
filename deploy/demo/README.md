# The whole agent, on AWS, for anyone to run

One model key and AWS credentials, four commands:

```sh
OPENAI_API_KEY=… ./deploy/demo/up.sh      # brings everything up, prints where the agent is
./deploy/demo/demo.sh                     # a scripted conversation: the checks, a run record, a report
./deploy/demo/demo.sh kill                # SIGKILL the agent mid-job; the session survives
./deploy/demo/down.sh                     # deletes everything, then proves it is gone
```

What runs is the real thing: `beyond-ai-agent serve --service` talking to a real model through the
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
- gets the same static `exec-stub` binary, uploaded through E2B's envd file API and started with
  envd's process API, listening on 8081, which E2B exposes at `https://8081-<sandbox id>.e2b.app`;
- is the exec endpoint of its session only: the session's grant carries that URL and a bearer token
  (an exec header, sealed in the grant), and the stub answers 401 to anything without it, since the
  endpoint is on the public internet.

Job-end notify across the internet: the agent task announces `https://<its-public-ip-dashed>.sslip.io:8443`
as its `--notify-origin` (computed at every start, so a replacement task announces its new address),
the demo's throwaway CA has a certificate for `*.sslip.io`, and each VM trusts that CA only. Port 8443
is open to the world in this mode; it accepts only a callback token the agent's fleet key minted.
`sslip.io` is a public wildcard DNS service that resolves `1-2-3-4.sslip.io` to `1.2.3.4`.

The E2B VMs have no EFS, so `sealed-at-rest` reads the session's bytes with a one-off Fargate task of
the agent's image that mounts the same access point. Two more checks join the list in this mode:
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
  names it.
- **browser**: the browser connector is the sandbox, and its address is the front too
  (`/mcp/<session>`). The agent offers the browser's tools at open from a list saved in the bundle
  (`.mcp-tools.json`, made at bundle build by `mcp-tools.sh` running `rustwright-mcp` once), so nothing
  dials the front and no VM exists. The model's first browser call reaches the front, which makes the
  VM, seeds it, installs the browser stack (the VM fetches the `rustwright-mcp` binary itself) and
  forwards; the agent then compares the server's real list with the saved one.
  `vm-created-on-first-tool` shows it, as for coding.

Every use case's report has a "Session open" section: sandbox requests while each session opened, and
the time from connecting to the first answer.

Before and after, per session, from the committed records (before: `20261020T230210Z-assistant`,
`20261020T231038Z-coding`, `20261020T231305Z-browser`; after: the records beside them from this change):

| use case  | sandbox requests at open, before | after         | open latency, before (derived) | after (measured)                                                                                                                       |
| --------- | -------------------------------- | ------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| assistant | 32                               | 0             | about 0.9 s                    | 302 ms (mean of 3)                                                                                                                     |
| coding    | 38                               | 0             | about 3.0 s                    | 306 ms, with no VM yet; the VM is made 2.7 s later, by the first tool call, and its first result comes 5.4 s after the create began    |
| browser   | 34 (and no others all run)       | 0 (0 all run) | about 3.0 s                    | 292 ms, with no VM yet; the first browser call makes the VM, installs the browser stack and returns its result 18.2 s after it started |

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

| check                   | what it shows                                                                                                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grant-refused`         | a tampered session grant is refused (401) at connect                                                                                                                        |
| `real-model-turn`       | a real model answers through the gateway; first-token latency, tokens                                                                                                       |
| `sandbox-exec`          | the model runs `uname -a; ls /` in the sandbox container, not on the agent                                                                                                  |
| `background-job-notify` | `sleep 20` runs in the background; the sandbox calls the agent back (`job_ended`) and the agent tells the model: the moment is printed, with the callback-to-notice latency |
| `notify-forged-refused` | a `job_ended` with the right kid and a wrong tag, sent while the session is open, is refused (401) and changes nothing                                                      |
| `kill-and-recover`      | a 140 s background job starts, the agent task is stopped (ECS SIGKILLs it), ECS replaces it, the same session reattaches from EFS and the job's end is reported             |
| `sealed-at-rest`        | the phrase you saw does not appear in the session's bytes on EFS; a hex excerpt is recorded                                                                                 |
| `billing-rows`          | every model call has exactly one `ai.usage` row, complete, and the cost adds up                                                                                             |
| `teardown-zero`         | filled in by `down.sh`: every resource type counted 0                                                                                                                       |

`demo.sh` runs the first seven and `billing-rows`; `demo.sh kill` runs `kill-and-recover` and
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
- **sandbox** task: `beyond-ai-demo exec-stub`, the repo's reference exec endpoint
  (`crates/test-support`), which passes the exec conformance suite, notify included
  (`crates/demo/tests/stub_conformance.rs`). It mounts the EFS read-only, which is how `sealed-at-rest`
  reads the bytes.
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
