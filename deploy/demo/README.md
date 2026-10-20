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
  sandbox's callback schedule retries until it does.
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
each of 20 resource types (SSM parameters by prefix, IAM roles by name, log groups, ECS clusters,
services, tasks and task definitions in every status, Cloud Map namespaces and services, the Route 53
zone, EFS file systems, mount targets and access points, the VPC, subnets, gateways, route tables,
security groups and network interfaces, ECR repositories) and the tagging API. It exits non-zero naming
anything that remains. ECS keeps deleted clusters, services and stopped tasks in the tagging index for
hours, so a tagging-API entry only counts if the service's own API cannot show it gone; the count that
was only index lag is reported beside it.
