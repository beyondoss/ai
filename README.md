# beyond/ai

Open-source AI at Beyond — a Cargo workspace:

- **`crates/gateway`** (`beyond-ai`) — the LLM egress gateway. Route LLM traffic through one internal proxy; apps use their stock OpenAI or Anthropic SDK unchanged — the gateway authenticates, swaps in the real provider key, and meters every token.
- **`crates/agent-core`** (`beyond-ai-agent-core`) — the agent harness core: a runtime/network-agnostic tool-calling loop (modeled on [pi](https://github.com/badlogic/pi-mono)), the OpenAI/Anthropic wire dialects, and the `ModelTransport`/`Tool` seams. Routes all model traffic through the gateway.
- **`crates/agent`** (`beyond-ai-agent`) — the agent CLI: `run` (one-shot coding task) and `serve` (headless control protocol over stdio, for remote control over SSH). Ships the coding tools (read/write/edit/bash/ls/grep/find) plus `todo` and `web`. Vendor-neutral: `--exec-url` points the whole toolset at any remote exec endpoint.

The gateway is the unified model API: the agent speaks OpenAI/Anthropic wire to it with a `bai_v1` key and never holds a provider key. See each crate's `ARCHITECTURE.md`.

## Gateway Quick Start

```sh
cp crates/gateway/config.example.toml config.toml
# Set at minimum: signing_keys and one pool key
AI_POOL_KEY_OPENAI=sk-... cargo run -p beyond-ai --release
```

Point any OpenAI-wire SDK at `http://ai.internal` with a virtual key:

```python
from openai import OpenAI
client = OpenAI(base_url="http://ai.internal/v1", api_key="bai_v1.1.<payload>.<sig>")
```

Or pass your own provider key directly (BYO — forwarded unchanged, no swap):

```python
client = OpenAI(base_url="http://ai.internal/v1", api_key="sk-your-openai-key")
```

### Using the Vercel AI SDK

`createOpenAI({ baseURL: "http://ai.internal/v1", apiKey })` works for every model. Its default
model speaks the Responses API and assumes the upstream stores every response, so it sends your
earlier assistant turns as `item_reference` ids instead of their text. Only OpenAI keeps those, and
the gateway stores no conversation content. On a non-OpenAI model (Claude, Llama, Grok, …) a
continued conversation is refused with a 400 naming `item_reference`; tool loops work as they are.
Send history in full with `store: false`:

```ts
const openai = createOpenAI({
  baseURL: "http://ai.internal/v1",
  apiKey: process.env.BAI_KEY,
});
await generateText({
  model: openai("claude-sonnet-5-5"),
  messages,
  providerOptions: { openai: { store: false } },
});
```

Or use `@ai-sdk/anthropic` (base URL `http://ai.internal/v1`) or `@ai-sdk/openai-compatible`, which
never send references. See `crates/gateway/ARCHITECTURE.md`, "Stored-response references".

## Agent Quick Start

**The gateway is optional.** Export a provider key and the agent goes straight to that provider — no
gateway, no extra config. The auth header (`x-api-key` for Anthropic, `Authorization: Bearer` for
everyone else) comes from the provider table; you never spell it out.

```sh
# One-shot coding task, straight to Anthropic
ANTHROPIC_API_KEY=sk-ant-... \
  cargo run -p beyond-ai-agent -- run --model claude-opus-4-8 "add a CHANGELOG entry"

# …or OpenAI, or any of the providers in the table (OPENROUTER_API_KEY, GROQ_API_KEY, …)
OPENAI_API_KEY=sk-... cargo run -p beyond-ai-agent -- run --model gpt-5 "…"

# A subscription login works too, with no gateway and no key at all
cargo run -p beyond-ai-agent -- login anthropic
```

Aggregators serve other vendors' model ids, so they can't be inferred from one — name them:

```sh
AI_PROVIDER=openrouter OPENROUTER_API_KEY=sk-or-... \
  cargo run -p beyond-ai-agent -- run --model anthropic/claude-sonnet-4.5 "…"

# Any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, a corporate proxy)
AI_BASE_URL=http://localhost:8000/v1 AI_API_KEY=local \
  cargo run -p beyond-ai-agent -- run --model qwen3-coder "…"
```

Point it at a gateway instead when you want key-swapping, routing, and metering — that path is
unchanged, and a configured gateway is never bypassed just because a key is in the environment
(`AI_DIRECT=1` overrides that, if you want it):

```sh
AI_GATEWAY_URL=http://ai.internal AI_AGENT_KEY=bai_v1... \
  cargo run -p beyond-ai-agent -- run "add a CHANGELOG entry for the latest release"

# Headless server — drive it over stdio (e.g. an SSH pipe); newline-delimited JSON
AI_GATEWAY_URL=http://ai.internal AI_AGENT_KEY=bai_v1... \
  cargo run -p beyond-ai-agent -- serve --session-file /tmp/agent.json
# then: {"type":"prompt","message":"…"} → streamed event frames, then a response

# Or over a loopback socket: WebSocket, or POST one command without holding a connection
cargo run -p beyond-ai-agent -- serve --listen 127.0.0.1:8787 --session-dir /tmp/sessions
curl -D- -H 'Content-Type: application/json' \
  --data '{"type":"prompt","message":"add a CHANGELOG entry"}' \
  'http://127.0.0.1:8787/_beyond/agent?session_id=build-42'
# 202 + X-Session-Id; the run continues. Watch it on a WebSocket to the same URL, or via --lifecycle-url.
```

**Credential precedence.** A gateway is _configured_ if `AI_GATEWAY_URL`, a stored `default_gateway_url`,
or `AI_AGENT_KEY` is set — the `http://ai.internal` default is a fallback, not configuration. With one
configured, the gateway is used (unless `AI_DIRECT=1`). Otherwise, routing is direct, in this order:

1. a `models.json` `base_url` override for the exact model id
2. `AI_BASE_URL` + `AI_API_KEY`
3. `AI_PROVIDER` + that provider's key
4. a stored OAuth login (`agent login`) — **above** an ambient key, so a stray `ANTHROPIC_API_KEY` in
   your shell never silently moves a subscription onto pay-per-token billing
5. the provider key for whichever provider natively serves the model id

Tools: `read`, `write`, `edit`, `bash`, `ls`, `grep`, `find` (pi's coding set), plus `todo` and `web`. See [crates/agent-core/ARCHITECTURE.md](crates/agent-core/ARCHITECTURE.md).

### Which session you get

A session id is an **address**, and it's the only thing that names a session outright. Both `run` and
`serve` take the same three:

```sh
agent run "..."                      # a new session, persisted under ~/.claude/sessions/<cwd>/
agent run --continue "..."           # reattach to this directory's most recent session
agent run --session-id build-42 "…"  # open session build-42, or create it — same session every time
```

`--session-id` outranks `--continue`, never resolves to a session it wasn't given, and is idempotent, so
it's the right flag for anything supervised or scripted: a restarted `serve --session-id X` lands back on
the same conversation deterministically, where `--continue` depends on whatever else touched the
directory meanwhile. Distinct ids in one directory are distinct sessions — which is what makes `serve`
multi-tenant, and what the daemon's `?session_id=` routing is built on.

A bare launch starts its own session rather than reattaching, so two shells (or two servers) working in
one directory don't silently drive the same transcript. Use `--continue` when you want the old one.

### Running the tools against a remote exec endpoint

`--exec-url <URL>` makes the agent's tools — `read`, `write`, `edit`, `ls`, `grep`, `find` **and
`bash`** — run wherever that URL points instead of on the host. The model sees an identical toolset:
same names, descriptions and JSON schemas, so no prompt changes and no prompt-cache miss.

The agent knows one protocol and nothing else. It does not know what is behind the URL, and it does
not manage sandboxes — creating, warming and destroying them is the caller's job.

```sh
AI_GATEWAY_URL=… AI_AGENT_KEY=… \
  cargo run -p beyond-ai-agent -- run \
    --exec-url https://exec.internal/run \
    --exec-header 'Authorization: Bearer $TOKEN' \
    "find the TODOs under /srv/app and fix the one in main.rs"
```

The protocol (v1.1) is one POST, deliberately the smallest thing that carries a command and its
result:

```jsonc
// request
{ "command": "rg", "args": ["--files"], "cwd": "/work", "timeout_ms": 120000 }
// request with stdin (optional): standard base64, written to the command's stdin, which is then closed
{ "command": "cat", "args": [], "timeout_ms": 10000, "stdin_base64": "c3RkaW4tcHJvYmU=" }
// response — 200
{ "exit_code": 0, "stdout": "…", "stderr": "…" }
```

Putting that in front of a provider's SDK is a few dozen lines on your side, and that shim is where
vendor specifics belong. `--exec-header` is repeatable and is where auth goes; which scheme the
endpoint wants is the endpoint's business.

`cwd` and `stdin_base64` are optional, and an endpoint that doesn't know `stdin_base64` (v1) keeps
working. The agent uses stdin only to write files, and only after a probe at attach proves it works:
it runs `cat` with stdin `stdin-probe`, and anything other than `stdin-probe` coming back means no
stdin. Without stdin, a write is sent as argv-sized chunks (48 KiB each, one command per chunk)
assembled in a temp file and renamed into place. Either way, a reader sees the old file or the new
one, never a partial one, and the size is checked before the rename, so a transport that drops stdin
fails the write instead of emptying the file. Size your endpoint's request limit for the files the
agent writes: with stdin, a request is about 4/3 of the file.

The agent reads at most 16 MiB of any one response (`--exec-max-response-bytes`). A bigger response
is an **error**, not a truncated result, because a cut-off JSON body can't be parsed honestly. The
error says the command ran, so the model narrows the output rather than re-running it. (That is
v1.1. A v1.2 stream, below, is read a frame at a time, so huge output gets the same bounded tail and
truncation marker as a local command.)

#### Streaming (v1.2): make `bash` behave as it does locally

Over v1.1, `bash` sees nothing until a command exits. It can't show progress, can't tell a silent
build from a hung one, and can't stop a command it has given up on. Locally, a command that is
silent for 60 s (or outlasts the model's `timeout_ms`) returns to the model as a numbered job that
`bash_job` can wait on or kill, and `run_in_background` starts a server or watcher as one. v1.2
makes the same true remotely, and keeps your endpoint stateless: **the connection is the command's
lifetime.** (Background jobs outlive connections anyway — see "Sandbox jobs" below.)

When the request carries `"stream": true`, answer `200` with `Content-Type: application/x-ndjson`
and one JSON frame per line:

```jsonc
{"stdout":"aGVsbG8K"}   // output bytes, standard base64, at most 64 KiB decoded per frame
{"stderr":"b29wcwo="}
{}                      // heartbeat: send one after 5 s with no other frame
{"shell_exit":0}        // the command exited, but processes of its group still run (not final)
{"exit_code":0}         // final, exactly one of: exit_code | signal | timed_out | error
```

- Write each frame as soon as you have it (you may coalesce up to 100 ms of output), and make sure
  nothing in front of you buffers the response.
- End the body right after the final frame. Use `{"signal":9}` for a signal, `{"timed_out":true}`
  when you enforce `timeout_ms` (kill the whole process group first), and `{"error":"…"}` if you
  fail after sending the 200. `bash` jobs have no deadline of their own, so a streamed `bash`
  command carries `"timeout_ms": 2147483647` (`i32::MAX`, the most a Node `setTimeout` holds): the
  connection ends it. Over v1.1, where nothing else could stop a hung command, `bash` sends 30
  minutes.
- **If the client disconnects before the final frame, you MUST kill the command's whole process
  group** (grandchildren too) within 10 s. Kill it at once if you can, by watching the connection.
  That is the only kill signal the agent sends: `bash_job` kill, a cancelled call and the agent's
  own exit all just close the stream.
- **If the command exits while its process group still has a live process** (`nohup server &`),
  you MUST send `{"shell_exit": N}` and keep the stream open until the group is empty, then send
  the command's own final frame. The agent tracks those processes as a job, exactly as it does a
  local shell's, and stops them by closing the stream. Zombies don't count.
- Keep answering requests without `stream` exactly as v1.1 does. The filesystem tools use them.
  Over v1.1 nothing can track what a command leaves running; kill its process group once you have
  answered.

The agent adopts streaming only after a probe at the first `bash` call proves it works: it runs
`printf a; sleep 0.3; printf b` and needs two frames, the first arriving well before the end. A v1.1
endpoint, or one whose frames arrive all at once, keeps v1.1 behavior: the call waits for the
command to exit, and `bash`'s description says so. The full spec is the module doc of
`crates/agent/src/exec_endpoint.rs`. To check your endpoint against it (framing, heartbeats,
timeouts, leftover processes, and kill-on-disconnect measured with a grandchild's marker file),
run:

```sh
AGENT_CONFORMANCE_EXEC_URL=https://exec.internal/run \
AGENT_CONFORMANCE_EXEC_HEADER='Authorization: Bearer …' \
  cargo test -p beyond-ai-agent --test exec_conformance -- --nocapture
```

The decoding vectors are in `crates/agent/tests/fixtures/exec/v1.2-frames.json`.

#### Sandbox jobs: background work that outlives the agent's replica

A job the model meant to keep — `run_in_background`, or a `server &` its command left running —
belongs to the **sandbox**, not to whichever agent replica started it: it runs until it exits, the
model or user kills it, or its session ends. A reaped, redeployed or failed-over replica only lets
go of it; the next replica to open the session lists the job, reports it still running under its
number, reads its output from where the model left off, and can kill it. A job that ended while no
replica was attached is reported once, as an ordinary finish notice.

**Your endpoint needs nothing for this.** The agent builds it out of ordinary commands — a directory
per job under `/tmp/beyond-jobs/`, a `setsid` supervisor, and a relay that keeps the last 8 MiB of
output in offset-named chunks — using `sh`, `setsid`, `mkfifo`, `tee`, `dd`, `wc`, `base64` and
`date`, all of which busybox has. It works over v1.1 too, so `run_in_background` works there (by
polling). Two things matter: **keep `/tmp` for the life of the sandbox**, and **let a process that
leaves the command's process group (`setsid`) outlive the request**. When the sandbox's lifetime
ends (you tear it down), its jobs end with it.

**v1.3 (optional): run jobs yourself.** If you would rather keep jobs your own way — your platform
has background commands with reconnectable handles, say — answer requests that carry a `"job"`
object. The agent probes with `{"command": "sh", "args": ["-c", "printf exec-v1.2"], …, "job":
{"op": "probe"}}`; answer `{"jobs": "v1.3"}`, and every job operation then comes to you:

```jsonc
// start, held: the job lives as long as this stream unless kept
{"command": "sh", "args": ["-c", "npm run dev"], "cwd": "/work", "stream": true,
 "job": {"op": "start", "scope": "9f2c…", "id": "4a7e…", "hold": true, "label": "npm run dev"}}
// → {"started": 4242} {"stdout": "…"} … {"shell_exit": 0}? … {"exit_code": 0}
{"command": "true", "args": [], "job": {"op": "keep", "scope": "9f2c…", "id": "4a7e…", "num": 1, "kind": "background", "seen": 120}}
{"command": "true", "args": [], "stream": true, "job": {"op": "attach", "scope": "9f2c…", "id": "4a7e…", "offset": 120}}
// → {"gap": n}? {"stdout": "…"} … {"exit_code": 0}
// read, kill, list, remove, wipe, seen: one JSON answer each
```

The full table — scopes, states, what `kill` must and must not signal, which frames a job stream
adds to v1.2's — is the "Sandbox jobs (v1.3)" section of `crates/agent/src/exec_endpoint.rs`. The
conformance run above checks it when you offer it (a dropped hold must kill an unkept job, a kept one
must survive and read back from an offset, `kill` must reach a job that ignores `SIGTERM`, and
`list`/`wipe` must stay inside their scope). The decoding vectors are in
`crates/agent/tests/fixtures/exec/v1.3-jobs.json`.

#### Multi-tenant `serve`

In `serve`, the endpoint is **per session**, so one server can multiplex tenants with a sandbox each:

```jsonc
{"id":"1","type":"set_exec_endpoint","url":"https://exec.internal/run","headers":["Authorization: Bearer …"]}
{"id":"2","type":"set_exec_endpoint","command":"docker exec tenant-a {}"}   // or a template
{"id":"3","type":"set_exec_endpoint"}                                        // detach, back to the host
```

Three properties this guarantees, each covered by a test:

- **Switching sessions never inherits an endpoint.** The incoming session is detached first, then
  reattached from its own record. Failing open to "runs on the host" is recoverable; failing open to
  "runs in the previous tenant's sandbox" is not.
- **Subagents act on their parent's machine.** A child whose tools ran on the host while its parent
  was sandboxed would let the model escape by delegating.
- **Resuming a session reattaches its endpoint**, warm or cold — switching back to it, and reopening
  it in a brand-new process — so a restart doesn't silently move a tenant's work onto the server. A
  fork does _not_ inherit it: a sandbox belongs to the session it was made for.

`--exec-url`/`--exec-cmd` also work on `serve` as a process-wide default for the single-tenant case.
It is a starting point, not a floor: a session with no endpoint of its own gets it, and a session
that has one keeps its own — whether it was set on this connection or recorded on a previous run.

For targets with no HTTP surface, `--exec-cmd` takes an argv template whose `{}` is replaced by the
command:

```sh
--exec-cmd 'ssh build-host -- {}'
--exec-cmd 'docker exec my-container {}'
--exec-cmd 'kubectl exec pod/app -- {}'
```

`{}` expands to **separate argv entries**, never into a shell string, so a path containing spaces,
quotes or `;` stays one argument on the far side and cannot be reparsed as syntax.

`ssh` forwards stdin, so file writes go in one command. `docker exec` and `kubectl exec` forward it
only with `-i` (`docker exec -i my-container {}`). Without `-i`, writes still work, but they fall
back to one command per 48 KiB.

A template's output streams, because the transport is a local process, so `bash` yields and shows
progress as it does locally. Killing the transport does **not** kill the command: a SIGKILL or
SIGTERM to `docker exec` leaves its command running in the container (measured). So every command
runs under a small `sh` wrapper that starts it in its own session (`setsid`) and reports its process
group on stderr (the agent strips that line). A kill (`bash_job`, a cancelled call, a timeout, the
agent's exit) then runs `kill -KILL -<pgid>` through the same template. The target needs `sh`, and
`setsid` for the kill to reach grandchildren (busybox and util-linux both have it). Without `setsid`,
only the command's own process is killed.

**Install `ripgrep` on the target if you can.** With `rg` present, results are byte-identical to
running locally. Without it the fallback is POSIX `grep`, which diverges in two measured ways: it
does not honor `.gitignore` (so it walks `target/` and `node_modules/`), and `grep -I` treats any
file containing non-UTF-8 bytes as binary and **silently skips it**. Both are asserted by name in
`crates/agent/tests/fs_backend_parity.rs`.

One thing to know: **paths are the target's.** Point the endpoint at a machine whose working tree is
at the same absolute path as the host's, or paths the agent reads from skills and the system prompt
won't resolve.

### Deploying a service-mode replica

`serve --service` is the multi-tenant deployment of the above: one replica, many tenants, every
connection carrying a verified session grant (`x-beyond-grant`) that says which tenant and session it
is, where its sandbox is, and — sealed — the credentials to reach them. Nothing of the replica's
reaches a tenant: not its `$HOME`, its settings, its cwd, or its filesystem. See
[crates/agent/ARCHITECTURE.md](crates/agent/ARCHITECTURE.md#service-mode--one-replica-many-tenants).

This section covers running **one** replica. For a fleet of them, see
**[crates/agent/FLEET.md](crates/agent/FLEET.md)**: what service mode is for, the **edge contract**
(routing obligations the agent cannot enforce — consistent hashing, the ring walk, a retry budget
longer than the storage lease), sizing and blast radius, what to alert on, how it scales, and what it
costs against the alternatives. Every number there was measured; `deploy/efs-proof/results/` holds
the run they came from.

`Dockerfile.agent` builds it — a static musl binary in an Alpine runtime, entrypoint
`serve --service`, non-root (uid 10001):

```sh
docker build -f Dockerfile.agent -t beyond-ai-agent .

docker run --rm -p 8080:8080 \
  -v /mnt/efs/a07-s1:/mnt/efs/a07-s1 \
  -v /mnt/efs/a07-s2:/mnt/efs/a07-s2 \
  -v /etc/beyond/grant:/etc/beyond/grant:ro \
  beyond-ai-agent \
  --listen 0.0.0.0:8080 \
  --gateway-url https://gateway.internal \
  --grant-key k1=/etc/beyond/grant/k1.pub \
  --seal-key /etc/beyond/grant/seal.key \
  --shard a07-s1=/mnt/efs/a07-s1 \
  --shard a07-s2=/mnt/efs/a07-s2
```

- **Required:** the grant verifier (`--grant-key`, repeatable, plus `--seal-key`), at least one
  `--shard <name>=</abs/path>`, and a listener (`--listen`, `--listen-uds`, or systemd activation).
  Stdio `--service` is refused, as are `--key`, `--session-dir`, `--exec-url` and the other flags
  that would make one tenant's setting everybody's — see ARCHITECTURE.md for the list.
- **Mounts:** one per shard at `/mnt/efs/<shard>`, each a directory writable by uid 10001. Tenant
  data lives at `<shard>/<tenant>/{sessions,memory}/`; a session id carries its shard as a
  `<shard>.` prefix, so an id alone says which mount serves it. Credentials mount read-only. A
  tenant's workspace is **not** here — it is in that tenant's sandbox, behind the exec endpoint in
  its grant.
- **Metrics:** `--metrics-listen 127.0.0.1:9095` serves `GET /metrics` on a listener of its own.
  Loopback addresses only — the scrape describes every tenant on the replica and the replica is
  reachable by tenants, so a routable bind is refused rather than warned about. Run the scraper
  beside it. No tenant or session identifier appears in any label.
- **Listings:** `list_sessions`/`list_all_sessions` take `query` (case-insensitive substring over
  `title`/`id`/`preview`/`cwd`), plus `limit` (default **50**, capped at **500**) and `offset`; the
  response carries `total` beside `sessions` so a client can page without guessing. Entries are
  metadata — a listing never carries the sessions' text.
- **Session titles:** a session names itself from its first successful run, via one extra model
  call, and never again. Best-effort throughout: an unusable or failed generation leaves the session
  untitled and working. The title is also announced on the `session_named` lifecycle event, which is
  how a catalog outside the fleet learns it.
- **Idle sessions:** `--session-idle-timeout` defaults to **60s in service mode** (an hour for the
  single-user daemon). A detached session holds its session lock until it is reaped, and on a replica
  that lock is what a failed-over session's new connections wait on — so the window is short, and
  being wrong costs a respawn from shared storage rather than availability.
- **Draining:** `--drain-grace <seconds>` makes SIGTERM stop taking _new_ sessions while finishing
  what is in flight — `/readyz` goes 503 so the load balancer stops choosing this replica, `/livez`
  stays 200 so the orchestrator doesn't kill it mid-drain, and reconnects to sessions this replica
  already owns keep working. Set `terminationGracePeriodSeconds` above it. `0` (default) shuts down
  at once.
- **Health**, on the same listener, no grant needed:

  ```sh
  curl -fsS http://127.0.0.1:8080/livez   # 200 once the process is serving
  curl -isS http://127.0.0.1:8080/readyz  # 200 ready, or 503 {"reason":"shard a07-s2: not writable"}
  ```

  Wire `/readyz` to the readiness probe and `/livez` to the liveness one, never the reverse:
  `/readyz` fails while a shard is missing or read-only, which should take the replica out of
  rotation — restarting it would not put the mount back. The image's `HEALTHCHECK` already hits
  `/readyz` (set `AGENT_PORT` if you bind anywhere but 8080).

## What It Does

- **Managed keys** (`bai_v1…` / `bai_v2…`): Ed25519-verified and stateless. The gateway swaps in the pool key, attributes usage to tenant and VPC, and checks the deny-set (spend/fraud) and the allowance-set (quota). They reach only `POST` generation endpoints: chat/completions, messages, responses, embeddings, and the count_tokens/compact sub-resources. On managed `/v1` and `/auto` the request walks the model catalog, with failover and translation between the OpenAI and Anthropic wires.
- **BYO keys**: any other token is forwarded to the provider unchanged. There is no key swap, deny-set, attribution or `ai.usage` billing event; aggregate throughput metrics still count it. A BYO caller never uses the catalog, and that includes `GET /v1/models`: it relays to the provider the key belongs to (the key's shape, else the path's default dialect) and returns that provider's own list.
- **`GET /v1/models` with a managed key lists what this deployment can serve**: the catalog rows with at least one candidate whose provider has a pool key configured. A row whose primary provider is unkeyed but whose fallback is keyed is still listed. The body is built once at boot. A BYO key gets its provider's own list (see above); nothing serves the full, unfiltered catalog.
- **12 providers, zero config**: openai, anthropic, openrouter, fireworks, groq, deepseek, together, cerebras, mistral, xai, bedrock and openai-codex (BYO only). Add more in `config.toml` under `[provider_authorities]`.
- **Streams responses**: SSE is relayed event by event and never held whole. A 64 KB tail taps usage. Request bodies are buffered where routing needs them: managed catalog walks (to rewrite `model`), usage injection on managed OpenAI streams, Responses, bodies over 64 KB, and translated non-stream responses (up to 32 MiB).
- **List prices on the catalog, token facts on the request**: `GET /v1/models` publishes a standard list price for every catalog model (USD per million tokens). `ai.usage` stays token counts (stdout → logfwd/OTLP → ClickHouse). A closed downstream consumer applies that card, or a contract rate. Slipstream (NATS) carries the deny-set, the allowance-set and the capture-set.
- **Guardrails**: a per-key request ceiling (`rate_limit_rps`), an optional per-tenant concurrency cap, and a per-provider circuit breaker that stops sending traffic to a provider returning 5xx or failing to connect. The deny-set and allowance-set own spend control.
- **NATS outages**: the deny-set fails open (a NATS outage stales it; existing allows stay allowed). The allowance-set fails closed: until it has been read once, from NATS or an on-disk snapshot, managed requests get 402 `allowance unavailable`. Run NATS (or seed `snapshot_path`) before serving managed traffic.

## Providers

The provider is the **first path segment** of the base URL — no header, nothing tool-specific. Bare
`/v1` defaults to OpenAI (and `/v1/messages` to Anthropic), so the two big providers are a host-only
swap; everything else is `/{provider}/…` using that provider's own path (forwarded verbatim).

```python
# OpenAI (default) — change only the host
client = OpenAI(base_url="http://ai.internal/v1", api_key="bai_v1...")

# Groq — its native base path is /openai/v1, so the gateway path is /groq/openai/v1
client = OpenAI(base_url="http://ai.internal/groq/openai/v1", api_key="bai_v1...")

# Fireworks mounts at /inference/v1 → /fireworks/inference/v1; OpenRouter at /api/v1 → /openrouter/api/v1
```

Bedrock's Anthropic Messages surface is `/bedrock/anthropic/v1/messages` — same SDK as Anthropic, different first segment.

An unknown first segment is a 404. See `providers::gateway_providers` (crates/providers/src/lib.rs) for each provider's native base path.

## Config

Scalar config keys are overridable by `AI_`-prefixed env vars (`AI_NATS_URL`, …), and pool and signing keys by `AI_POOL_KEY_<NAME>` / `AI_SIGNING_KEY_<KID>`. Map keys (`provider_authorities.*`, `provider_dialects.*`, `provider_auth_schemes.*`) are config-file only. See `crates/gateway/config.example.toml` for the full reference.

Required to serve managed traffic:

| Key                  | Source        | Purpose                                                 |
| -------------------- | ------------- | ------------------------------------------------------- |
| `signing_keys`       | `config.toml` | Ed25519 public keys by `kid` — verifies `bai_v1` tokens |
| `AI_POOL_KEY_<NAME>` | env (SSM)     | Provider key swapped in for managed requests            |

Optional:

| Key                      | Default   | Purpose                                                                  |
| ------------------------ | --------- | ------------------------------------------------------------------------ |
| `snapshot_path`          | unset     | On-disk deny-set snapshot — set on durable nodes, leave unset on Fargate |
| `rate_limit_rps`         | `100`     | Per-key request ceiling; `0` disables                                    |
| `[provider_authorities]` | built-ins | Override or add upstream hosts                                           |

## Running Tests

```sh
mise run test:unit:rs        # pure-logic unit tests (no network)
mise run test:integration:rs # gateway + mock upstream + NATS
mise run test:smoke          # live providers — needs API keys in env, bills real (tiny) requests
mise run bench               # unit micro-benchmarks + end-to-end throughput
```

## Architecture

[crates/gateway/ARCHITECTURE.md](crates/gateway/ARCHITECTURE.md) — request flow, module map, key invariants.

## Third-Party Notices

The agent crates are modeled on [pi](https://github.com/badlogic/pi-mono) (MIT, Mario
Zechner). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its license text.
