# agent

A coding agent in one static binary. It reads a task, calls a model, and runs
tools (file edits, search, shell) until the task is done.

**Run the agent anywhere, in front of anyone, and it keeps the session.** The
hard parts of running agents in production (durable sessions, scale, security,
portability) are built, proven on real infrastructure, and open.

|                                            | The fact                                                                                                                                                                                                                                                                                                                               | The proof                                                                                                                                                                           |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Any pattern**                            | The same binary runs _inside_ the sandbox (local tools, one process, nothing to operate) or _outside_ it: `serve --service` holds the conversation and sends every tool call to a remote sandbox over `--exec-url`. Outside, keys never enter the sandbox, the sandbox can sleep or die, and one replica serves thousands of sessions. | [Service mode](ARCHITECTURE.md#service-mode--one-replica-many-tenants), [exec endpoint](#run-the-tools-somewhere-else)                                                              |
| **Sessions that survive a killed replica** | Kill a replica mid-job and another picks the session up from the shared store; an epoch fence on the store keeps two replicas from ever writing one session. On real ECS and EFS no acknowledged write was lost across 142,755 turns through rolling deploys and hard kills, and the network-partition takeover passes too.            | [`deploy/efs-proof`](../../deploy/efs-proof/README.md), [the epoch fence](ARCHITECTURE.md#ownership-the-epoch-fence), `demo.sh kill` ([`deploy/demo`](../../deploy/demo/README.md)) |
| **Work outlives the agent**                | A kept or background job runs in the sandbox, not the replica. When it ends, the sandbox calls back (job-end notify) so the finish is reported to the session, even if the replica that started it is gone.                                                                                                                            | [`exec-notify`](../exec-notify/README.md), [Run the tools somewhere else](#run-the-tools-somewhere-else)                                                                            |
| **A connection is only a view**            | Disconnect, reattach from another device, or watch from several at once. The session keeps running and a late attach is caught up server-side.                                                                                                                                                                                         | [Serve](#serve), [`serve`](ARCHITECTURE.md#serve--headless-ndjson-control-protocol)                                                                                                 |
| **Any medium**                             | One protocol over stdio (built for an `ssh` pipe), WebSocket, HTTP POST, a Unix socket, or TLS.                                                                                                                                                                                                                                        | [Serve](#serve)                                                                                                                                                                     |
| **The agent never holds a key**            | Provider keys stay in the gateway (`bai_v1` virtual key), connector credentials in the broker (`bmc_v1`), and a grant's secrets are sealed to the replica. In the demo the sandbox cannot open a TCP connection to the session store.                                                                                                  | [Service mode](ARCHITECTURE.md#service-mode--one-replica-many-tenants), [`BEYOND.md`](BEYOND.md), demo check `sandbox-cannot-reach-store`                                           |
| **Ciphertext at rest, per tenant**         | Every transcript line is XChaCha20-Poly1305 under the tenant's key, bound to tenant and session id. Destroying the tenant's key crypto-erases its sessions.                                                                                                                                                                            | [Sealing](ARCHITECTURE.md#sealing), demo check `sealed-at-rest`, [`BEYOND.md`](BEYOND.md)                                                                                           |
| **Dense and cheap**                        | One 0.5 vCPU, 1 GB Fargate replica held about 2,900 live sessions per wave with 0 failed opens and 520 MB of peak anonymous memory (about 170 KB per session) against a scripted model. Idle sandboxes can sleep, since nothing depends on them being up. A freshly started `serve` idles at 5.3 MB RSS.                               | [`deploy/demo/load.sh`](../../deploy/demo/load.sh), [`FLEET.md`](FLEET.md#how-it-scales), [#50](https://github.com/beyondoss/ai/pull/50)                                            |
| **Any model, any sandbox**                 | OpenAI, Anthropic, Groq, OpenRouter, Bedrock and more, direct or through the gateway, and a session resumes across providers mid-conversation. One exec protocol reaches Beyond, E2B, SSH, Docker or your own sandbox, and a conformance kit tells you whether yours speaks it.                                                        | [`providers`](../providers/README.md), [`BEYOND.md`](BEYOND.md), [exec conformance](#run-the-tools-somewhere-else)                                                                  |
| **A full coding harness**                  | File, shell and background-job tools, an MCP client, skills, memory, compaction, approval gates.                                                                                                                                                                                                                                       | [Tools](#tools)                                                                                                                                                                     |

Every row has a runnable check: `mise run sim:matrix` drives a replica set
against a shared filesystem and `deploy/demo/demo.sh` runs the live
checks against a real AWS stack (or E2B). The sandbox filesystem backend is
tested against a randomized differential model, not just examples.

Three ways to run it:

| Command           | Use                                                                        |
| ----------------- | -------------------------------------------------------------------------- |
| `run "<task>"`    | One task, output streamed to stdout, exit when done.                       |
| `serve`           | A JSON-lines control protocol: stdio, WebSocket, HTTP POST or Unix socket. |
| `serve --service` | One replica serving many tenants, each connection carrying a signed grant. |

The agent needs a model. It reaches one directly with your provider key, or
through a [Beyond AI gateway](../gateway/README.md) with a virtual key. It
runs Linux only.

## Install

Static `linux-amd64` and `linux-arm64` binaries are attached to each
[GitHub release](https://github.com/beyondoss/ai/releases):

```sh
V=<tag>   # a release cut after the binary was renamed to `agent`
curl -fsSL "https://github.com/beyondoss/ai/releases/download/$V/agent-$V-linux-amd64.tar.gz" \
  | tar xz
./agent --version
```

From a checkout, `cargo run -p beyond-agent -- <args>` runs the same binary.

## Try it

Export a provider key. The agent picks the provider from the model id and
sends the key in the header that provider expects.

```sh
ANTHROPIC_API_KEY=sk-ant-... agent run "add a CHANGELOG entry"
OPENAI_API_KEY=sk-...        agent run --model gpt-5 "explain src/main.rs"
agent list-models   # known model ids and their capabilities
```

The default model is `claude-opus-4-8`; `--model`, `AI_AGENT_MODEL` or
`settings --model` change it.

| You have                                    | Set                                                                                                              |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| A provider key (OpenAI, Anthropic, Groq, …) | The provider's variable, for example `ANTHROPIC_API_KEY`                                                         |
| A subscription                              | `agent login anthropic` (or `github-copilot`, `openai-codex`)                                                    |
| An aggregator key (OpenRouter, Together, …) | `AI_PROVIDER=openrouter` and its key; aggregators serve other vendors' model ids, so the agent cannot infer them |
| An OpenAI-compatible server (vLLM, Ollama)  | `AI_BASE_URL=http://localhost:8000/v1 AI_API_KEY=local`                                                          |
| A Beyond AI gateway                         | `AI_GATEWAY_URL=http://ai.internal AI_AGENT_KEY=bai_v1...`                                                       |

The provider variables are listed in
[`providers`](../providers/README.md#providers). A configured gateway is
never bypassed because a provider key is also set; `AI_DIRECT=1` overrides
that. The order in which these are tried when several are set is in
[Provider routing](ARCHITECTURE.md#provider-routing--the-gateway-is-optional).

A model run needs a live provider. To see the control protocol without one,
set a dummy key and ask for state:

```sh
echo '{"id":"1","type":"get_state"}' \
  | ANTHROPIC_API_KEY=dummy agent serve --session-file /tmp/agent.json
# {"type":"ready",...} then {"command":"get_state","data":{...},"success":true,...}
```

## Tools

| Tool                                          | Offered                                                                                                                                                                                                              |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `read`, `write`, `edit`, `ls`, `grep`, `find` | Always. `grep` and `find` honor `.gitignore`.                                                                                                                                                                        |
| `bash`, `bash_job`                            | Always. A command silent for 60 s, or past its `timeout_ms`, returns as a numbered job that `bash_job` waits on or kills; `run_in_background` starts a server or watcher as one. The model is told when a job exits. |
| `todo`                                        | Always.                                                                                                                                                                                                              |
| `web`                                         | Only when named: `--tools read,bash,web`. Fetches, converts to markdown, extracts rows. Refuses loopback and private addresses unless `--web-allow-private` or `--web-allow-host`.                                   |
| `memory`                                      | Unless `--no-memory`. Durable notes, per project.                                                                                                                                                                    |
| `subagent`                                    | When an agent definition exists (`.claude/agents/*.md`, `~/.claude/agents/*.md`): runs a nested agent in-process, optionally in a git worktree.                                                                      |
| `structured_output`                           | With `run --output-schema <schema>`: the last stdout line is JSON matching the schema.                                                                                                                               |
| `mcp__<server>__<tool>`                       | One per tool of each MCP server in `mcp_servers` or `.mcp.json`.                                                                                                                                                     |

`agent tools` prints the default set with schemas. `--tools`,
`--exclude-tools`, `--no-tools`, `--deny-tool`, `--deny-bash-pattern` and
`--deny-path` narrow what the model may call.

Around the tools:

- **MCP client.** stdio and HTTP servers, OAuth (`mcp-login <name>`), tool
  search for large catalogs, MCP Apps, MCP Events. See
  [MCP tool names](ARCHITECTURE.md#mcp-tool-names) and
  [`.mcp.json`](ARCHITECTURE.md#mcpjson).
- **Skills, prompt templates and `AGENTS.md`/`CLAUDE.md`** are read from the
  project and from `~/.claude`. An `AGENTS.md` may open with frontmatter that
  picks the model and tools
  ([`AGENTS.md` frontmatter](ARCHITECTURE.md#agentsmd-frontmatter--the-agents-own-definition)).
- **Compaction.** When the prompt nears the context window, old tool output is
  masked and older turns are summarized. `--no-compaction` turns it off.
- **Approval.** `serve --approve writes|all|tools:<name>` asks attached clients
  before running tools
  ([Interactive approval](ARCHITECTURE.md#interactive-approval--the-human-in-the-loop-gate)).
- **Trust.** A project's `.claude/SYSTEM.md` and settings apply only after
  `agent trust` (or `-a` for one run). Trust gates which instructions
  the model sees, not what it can do.

## Sessions

A session id is an address. `run` and `serve` take the same three forms:

```sh
agent run "..."                       # a new session
agent run --continue "..."            # this directory's latest session
agent run --session-id build-42 "..." # open build-42, or create it
```

`--session-id` outranks `--continue` and is idempotent, so a restarted
`serve --session-id X` lands on the same conversation. A bare launch always
starts its own session, so two shells in one directory do not share a
transcript. Sessions are JSON-lines files under `~/.beyond/sessions/<cwd>/`;
`--session-dir` moves them, `--no-session-persistence` skips them, and
`export <file.jsonl>` renders one as HTML.

## State

The agent writes only under `~/.beyond` (or `$AI_AGENT_CONFIG_DIR`): settings,
logins, trust grants, `models.json`, memory, the MCP cache and sessions. A
project's own settings are `<project>/.beyond/settings.json`. `~/.claude`
belongs to Claude Code; the agent reads skills, agents, prompt templates and
`CLAUDE.md` from it and never writes there. The file table is in
[Where state lives](ARCHITECTURE.md#where-state-lives--beyond-and-claude-read-only).

## Serve

```sh
# stdio: one JSON command per line in, one JSON frame per line out
agent serve --session-file /tmp/agent.json
# {"type":"prompt","message":"..."} -> ack, event frames, then a response

# loopback socket: WebSocket (GET) and one-shot commands (POST)
agent serve --listen 127.0.0.1:8787 --session-dir /tmp/sessions
curl -D- -H 'Content-Type: application/json' \
  --data '{"type":"prompt","message":"add a CHANGELOG entry"}' \
  'http://127.0.0.1:8787/_beyond/agent?session_id=build-42'
# 202 + X-Session-Id: the run continues without the connection
```

One `serve --listen` process holds many sessions, one per `session_id`. A
POST returns the command's `ack` (a `prompt` is 202) or `response` (200); a
WebSocket on the same id sees the same session. `--listen-uds <path>` serves
the same protocol on a Unix socket whose file mode decides who connects, and
`--tls-listen` with `--tls-cert` and `--tls-key` adds TLS. `--lifecycle-url`
POSTs `started`, `progress` and `finished` events for each prompt. The command
and frame reference is the module doc of `src/serve/mod.rs` and
[`serve`](ARCHITECTURE.md#serve--headless-ndjson-control-protocol).

## Run the tools somewhere else

`--exec-url <URL>` sends every tool call, `bash` included, to a remote endpoint instead of running it on the host. The model
sees the same tools, with the same schemas and descriptions, so the prompt cache stays warm. The agent knows one protocol; it
does not create or destroy sandboxes.

**Put [`execd`](../execd/README.md) in the sandbox image and you are done.** It is the endpoint, already written:

- **Conformant.** Exec protocol (one-shot, streaming with kill on disconnect, sandbox-native jobs, job-end notify): 32/32
  conformance checks live on AWS Fargate and on E2B.
- **Small.** One 5 MB static binary (linux amd64 and arm64), about 2,372 KiB idle PSS, zero idle wakeups.
- **Isolated.** Under `EXECD_RUN_AS` the commands run as another user and cannot read execd's token or memory.
- **Fails closed.** Without a bearer token it refuses to start on any address but loopback.

Every recipe below is proven. Pick the one that matches how you build the sandbox.

**Dockerfile.** The image is `FROM scratch` with the binary at `/execd`:

```dockerfile
COPY --from=ghcr.io/beyondoss/execd:v0.1.11 /execd /usr/local/bin/execd
ENV EXECD_LISTEN=0.0.0.0:8080 EXECD_BEARER_FILE=/run/execd-bearer EXECD_RUN_AS=sandbox
CMD ["/usr/local/bin/execd"]
```

Write the token to `/run/execd-bearer` (root-owned, mode 0600) at start, and create the `sandbox` user the commands run as.

**Any image with a shell.** The installer checks the SHA-256 and is verified in Alpine and Debian:

```sh
curl -fsSL https://github.com/beyondoss/ai/releases/latest/download/install-execd.sh | sh
```

**E2B.** Upload the binary and a bearer file, then start execd as root with `EXECD_RUN_AS=user`. E2B's VM has passwordless
`sudo` and terminates TLS on `https://<port>-<sandbox id>.e2b.app`, so there is nothing else to run
([`deploy/demo/e2b.sh`](../../deploy/demo/e2b.sh) is the working script):

```sh
sudo chown root:root ~/execd ~/bearer && sudo chmod 600 ~/bearer
sudo env EXECD_LISTEN=0.0.0.0:8081 EXECD_ROOT=~/workspace EXECD_HOME=/home/user \
  EXECD_RUN_AS=user EXECD_BEARER_FILE=~/bearer setsid nohup ~/execd >/tmp/execd.log 2>&1 &
```

**Point the agent at it:**

```sh
agent run \
  --exec-url https://8081-<sandbox id>.e2b.app/ \
  --exec-header "Authorization: Bearer $TOKEN" \
  "fix the failing test under /srv/app"
```

`--exec-url` and `--exec-header` also work with `serve`. The agent probes the endpoint at attach and uses the highest version
it finds. An endpoint's response is capped at 16 MiB (`--exec-max-response-bytes`; keep `EXECD_MAX_RESPONSE_BYTES` equal to
it); a larger one is an error, not a truncation.

**Security, in three lines.** Run execd as root with `EXECD_RUN_AS`. Give it a bearer file. Put TLS in front of it, or use the
provider's (E2B does). What each control guarantees is in
[execd's ARCHITECTURE.md](../execd/ARCHITECTURE.md#security).

**Configuration** is `EXECD_*` environment variables only. [The table is in execd's README](../execd/README.md#configuration).

### Implementing your own endpoint

The spec is the module doc of `src/exec_endpoint.rs`; the prose is
[Running the tools against a remote exec endpoint](ARCHITECTURE.md#running-the-tools-against-a-remote-exec-endpoint). The base
protocol is one `POST`:

```jsonc
// request; cwd and stdin (standard base64, written to stdin) are optional
{ "command": "rg", "args": ["--files"], "cwd": "/work", "timeout_ms": 120000 }
// response, 200: stdout and stderr are the command's bytes, standard base64
{ "exit_code": 0, "stdout": "...", "stderr": "..." }
```

The same request with `"stream": true` streams output as NDJSON frames and treats a closed connection as the kill signal;
the endpoint owns background jobs so they outlive an agent replica; and it calls back when a kept job ends
([`exec-notify`](../exec-notify/README.md)). An endpoint implements all of it: nothing is probed, and the conformance suite
below grades the whole contract.

Try it locally with execd, no model needed. In one terminal:

```sh
mkdir -p /tmp/ws /tmp/ws-home
EXECD_LISTEN=127.0.0.1:8081 EXECD_ROOT=/tmp/ws EXECD_HOME=/tmp/ws-home \
  cargo run -p beyond-execd
```

In another, run `bash` through it with a `serve` command (the dummy key only satisfies startup):

```sh
(echo '{"id":"1","type":"bash","command":"pwd"}'; sleep 3) \
  | ANTHROPIC_API_KEY=dummy agent serve --exec-url http://127.0.0.1:8081/
# ... "result":"/tmp/ws\n" ...
```

For a target with no HTTP surface, `--exec-cmd` takes an argv template; `{}`
becomes separate argv entries, never a shell string:

```sh
--exec-cmd 'ssh build-host -- {}'
--exec-cmd 'docker exec -i my-container {}'
```

Install `ripgrep` on the target: without it `grep` falls back to POSIX `grep`,
which ignores `.gitignore` and skips files holding non-UTF-8 bytes. Paths are
the target's, so skills and the system prompt resolve only if the working tree
sits at the same absolute path.

In `serve`, an endpoint is per session: `set_exec_endpoint` attaches one,
a session switch never inherits the previous session's, and a resumed session
reattaches its own.

Check an endpoint against the protocol (framing, heartbeats, kill on
disconnect, leftover processes, job scopes):

```sh
AGENT_CONFORMANCE_EXEC_URL=https://exec.internal/run \
AGENT_CONFORMANCE_EXEC_HEADER='Authorization: Bearer ...' \
  cargo nextest run -p beyond-agent -E 'test(/^exec_conformance::/)' --no-capture
```

[`BEYOND.md`](BEYOND.md) lists the other conformance kits (exec front,
connector broker, bundles) and the contract a hosting platform implements.

## Service mode

`serve --service` is the multi-tenant deployment. Every connection carries a
verified `x-beyond-grant` naming the tenant, the session, its sandbox and,
sealed, the credentials to reach them. Nothing of the replica's reaches a
tenant: not its `$HOME`, settings, cwd or filesystem. The mechanism is
[Service mode](ARCHITECTURE.md#service-mode--one-replica-many-tenants); running
many replicas (routing, sizing, alerts, cost) is [`FLEET.md`](FLEET.md).

`Dockerfile.agent` builds a static musl binary in an Alpine image (non-root,
uid 10001, entrypoint `serve --service`). Multi-arch images are published as
`ghcr.io/beyondoss/ai-agent`, tagged `latest` and by commit SHA.

```sh
docker build -f Dockerfile.agent -t beyond-ai-agent .
docker run --rm -p 8080:8080 \
  -v /mnt/efs/a07-s1:/mnt/efs/a07-s1 \
  -v /etc/beyond/grant:/etc/beyond/grant:ro \
  beyond-ai-agent \
  --listen 0.0.0.0:8080 \
  --gateway-url https://gateway.internal \
  --grant-key "1=$(cat /etc/beyond/grant/k1.pub)" \
  --seal-key /etc/beyond/grant/seal.key \
  --shard a07-s1=/mnt/efs/a07-s1
```

| Flag                              | Meaning                                                                                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--grant-key <kid>=<base64>`      | Ed25519 public key that signs grants; `kid` is a decimal integer. Repeatable.                                                                                    |
| `--seal-key <file>`               | X25519 secret that opens a grant's sealed secrets. `AI_AGENT_SEAL_KEY_B64` carries it as an env secret instead; not both.                                        |
| `--seal-key-previous`             | The old key during a rotation.                                                                                                                                   |
| `--shard <name>=</abs/path>`      | A tenant-data mount; at least one. Data lives at `<shard>/<tenant>/{sessions,memory}/`.                                                                          |
| `--listen`, `--listen-uds`        | A listener is required; stdio `--service` is refused, as are flags that would make one tenant's setting everybody's (`--key`, `--session-dir`, `--exec-url`, …). |
| `--metrics-listen 127.0.0.1:9095` | `GET /metrics` on its own listener. Loopback only; no tenant or session id appears in a label.                                                                   |
| `--session-idle-timeout`          | Default 60 s in service mode: a detached session releases its lock fast so a failed-over session is not kept waiting.                                            |
| `--drain-grace <secs>`            | On SIGTERM, finish in-flight sessions and take no new ones; `/readyz` answers 503 while `/livez` stays 200.                                                      |
| `--max-live-sessions`             | Refuse new sessions past this count. Default 20000; `0` is no limit.                                                                                             |
| `--notify-origin`                 | Where sandboxes report a kept job's end (job-end notify).                                                                                                        |

`GET /livez` answers 200 once the process serves. `GET /readyz` answers 200,
or 503 with a reason (`draining`, or a shard that is missing or not writable,
checked by writing). Point the readiness probe at `/readyz` and the liveness
probe at `/livez`; the image's `HEALTHCHECK` uses `/readyz` on `AGENT_PORT`
(default 8080).

`list_sessions` and `list_all_sessions` take `query`, `limit` (default 50,
at most 500) and `offset`, and return `total` with `sessions`. Listings carry
metadata, never session text.

## Reference

- [`ARCHITECTURE.md`](ARCHITECTURE.md): how the agent works and why.
- [`FLEET.md`](FLEET.md): running service-mode replicas as a fleet.
- [`BEYOND.md`](BEYOND.md): what a hosting platform must provide, with the
  conformance kits.
- [`beyond-agent-core`](../agent-core/README.md): the loop this crate
  builds on.
- `agent <command> --help`: every flag, with its `AI_AGENT_*`
  variable.

The agent is modeled on [pi](https://github.com/badlogic/pi-mono); see
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
