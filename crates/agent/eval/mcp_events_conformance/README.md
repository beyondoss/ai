# MCP Events conformance (SEP-3415)

An end-to-end check of the agent's MCP Events client against an **independent** server. The server
(`server.py`) was written only from the spec text, in a different language from the client, so the two
cannot share a misreading. The client's own tests run against a fixture server written to the same draft
as the client; only this kit can see spec drift.

| File               | What                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| `server.py`        | A strict SEP-3415 server: Streamable HTTP and stdio, every delivery mode, a control endpoint.       |
| `selftest.py`      | Drives `server.py` with a raw client (no agent) to prove each server behaviour the check relies on. |
| `check.py`         | Drives the real `agent serve` against `server.py` and judges the agent.                             |
| `requirements.txt` | Exact versions.                                                                                     |

## Pinned sources

All of modelcontextprotocol/modelcontextprotocol at `bbc31f8ded5762eae4df115b93942f5e71c8b3a7`:

- `seps/3415-events-extension.md` (SEP-3415, Final) and `docs/extensions/events/`.
- `docs/specification/2026-07-28/`: Streamable HTTP (`Mcp-Method`, `Mcp-Name`, `Mcp-Param-*`,
  `x-mcp-header`, `HeaderMismatch` -32020, the Base64 sentinel), stdio, `server/discover`,
  `subscriptions/listen`, versioning and extension negotiation.

Also: modelcontextprotocol/ext-server-card (`Remote.extensionsInfo`, merged via ext-server-card#55) for the
signing-key location, and [Standard Webhooks](https://github.com/standard-webhooks/standard-webhooks)
(signed with the official `standardwebhooks` library). The normative text of
modelcontextprotocol/ext-events `specification/beta/events/events.mdx` was diffed against the SEP's
Specification section: identical apart from link targets and tense ("requests" vs "uses" the next five
codes). Nothing to reconcile.

## Run it

```sh
mise run eval:mcp-events-conformance            # builds the agent, installs the pinned env, runs both
mise run eval:mcp-events-conformance http       # or: stdio, all; --only SUBSTRING picks scenarios
```

By hand:

```sh
python3 -m venv venv && venv/bin/pip install -r crates/agent/eval/mcp_events_conformance/requirements.txt
cargo build -p beyond-agent
venv/bin/python crates/agent/eval/mcp_events_conformance/selftest.py
AGENT_BIN=target/debug/agent venv/bin/python crates/agent/eval/mcp_events_conformance/check.py all
```

Each assertion prints `PASS` or `FAIL` with a reason, and the exit status is nonzero on any `FAIL`. Nothing is
skipped: a missing dependency is a `FAIL` with the install line, and an assertion that cannot run because an
earlier step failed is itself a `FAIL`. Scenarios run four at a time (`CONFORMANCE_PARALLEL`). Per-run
state, `requests.log` and `violations.log` are under `/tmp/evconf-*`.

The agent needs a model to react to an event. `agent serve --gateway-url` points at a mock gateway in
`check.py` that speaks the Anthropic Messages streaming API and records each request body, so "the event
reached a model prompt" is asserted on the request the model received (the subscriptions use
`action: follow_up`), not on an agent frame. The agent reads `mcp_servers` from
`$HOME/.beyond/settings.json`; the check writes one per scenario under a throwaway `HOME`.

## What `check.py` asserts

Each scenario is a fresh server, agent and WebSocket. `{http,stdio} x {poll,push,webhook}` are one flow each.

| Assertion                                                                                                                                                                                                                                                                          | Where                 |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| (a) each mode delivers an emitted event as an `mcp_event` frame and into a model prompt; a non-matching argument is not delivered                                                                                                                                                  | every flow            |
| (b) HTTP `events/*` carry `Mcp-Name` equal to `params.name`; `x-mcp-header` arguments are mirrored as `Mcp-Param-*` (plain string, non-ASCII string in the Base64 sentinel, integer)                                                                                               | http flows            |
| (c) after `add event type` the client re-lists; it opted into `eventsListChanged` on `subscriptions/listen`                                                                                                                                                                        | every flow            |
| (d) `schema_changed` (`Unsupported`, poll: a stale schema-epoch cursor) leads to a re-list, a successful resubscribe and resumed delivery                                                                                                                                          | every flow            |
| (e) a revoke (`Forbidden` -32024) is surfaced as a refusal/termination, and retries back off (per request kind at most 6 attempts in 10 s, at most 3 in the next 10 s; `events/list` is made once per subscription attempt, so its limits scale by the subscription count)         | every flow            |
| (f) a server without the capability receives zero `events/*`                                                                                                                                                                                                                       | `*/no-capability`     |
| (g) a delivery with `v1` and a `v1a` signed by the Server Card key is delivered; the agent fetched the card; a correct `v1` with a `v1a` from another key is rejected 401                                                                                                          | `http/webhook-v1a`    |
| (h) after the server is killed and restarted with an event emitted while it was down, the client resumes from its persisted cursor, gets the event once, and nothing twice                                                                                                         | `http/*-restart`      |
| (i) after `add_tool` the client re-lists (`tools/list`) and the tool is in the agent session's own tool list (`get_mcp` frame); (ii) the new resource is in the agent session by next use (the spec does not mandate an immediate `resources/list`; the check prints it as `INFO`) | `*/core-list-changed` |
| (iii) the client opens `subscriptions/listen` and asks only for list-changed types the server advertised (a request for an undeclared type is logged; SHOULD-level, the spec has no MUST here); the server never sends a type that was not requested (`selftest.py`)               | `*/core-list-changed` |
| poll honours `nextPollMs`                                                                                                                                                                                                                                                          | poll flows            |
| `violations.log` is empty                                                                                                                                                                                                                                                          | every flow            |

(g) is **beyond SEP (our fail-closed policy)**, not a spec MUST, and is labelled so in the output. (iii) is beyond SEP too: SHOULD-level at most (the spec has no MUST on what a client may request). The SEP says "Endpoints that do not enforce server identity ignore
`v1a,`" and makes `v1a` optional for the server; the project requires the agent to enforce it when the
Server Card publishes a key. Over stdio there is no MCP origin and so no Server Card: the server signs `v1`
only and (g), (h) are HTTP-only. (h) over stdio is not asserted: restarting a stdio server is a client
SHOULD ("the client SHOULD restart it").

## How strictness works

`server.py` performs every server-side MUST, and checks every client-side MUST it can observe. A client
violation is answered with the spec's error **and** appended to `violations.log` (JSON lines: rule,
detail), which `check.py` requires to be empty. `requests.log` records every request (method, name, cursor,
`Mcp-*` headers, outcome) for the counts. `selftest.py` asserts the server's own behaviour (46 checks).

The `mcp` SDK (2.3.0) implements none of `events/*`, `subscriptions/listen` with `eventsListChanged`, or
the webhook machinery, so the JSON-RPC envelope, SSE framing and every events method are hand-rolled on
Starlette. It is pinned because the interop harness next door uses it; this kit imports none of it.

### Control endpoint (`POST /control`, local only)

`emit {type,data,event_id}`, `add_type`, `remove_type {name}`, `schema_change {name}`, `revoke`, `restore`,
`bad_v1a` (a delivery with a good `v1` and a rogue `v1a`), `state {since}` (subscriptions, open streams,
listeners, requests). In stdio mode the control port is `--control-port`.

### Flags

`--allow-http-callback` is **for local testing only**: the SEP requires an `https` callback (`InvalidParams`
otherwise) and the check's agent receives webhooks on loopback over `http`. Without the flag the server
refuses non-`https` URLs and non-globally-routable addresses. `--no-events-capability`, `--page-size`
(`events/list` pagination, default 2), `--poll-ms`, `--heartbeat-ms`, `--ttl-ms`, `--cursor-null-in-events`,
`--emit-on-start` (the restart scenario). The server needs at least one tool for the agent to keep it
connected, so it offers `echo`.

## Every SEP MUST the server checks

"Server" rows are obligations on the server that it performs; "Client" rows are obligations on the client
that it checks and logs.

| MUST                                                                                                                                                                                                       | Section                                                                | Where checked                                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Server declares `capabilities.extensions["io.modelcontextprotocol/events"]` with `listChanged`                                                                                                             | SEP-3415 Capability Negotiation                                        | `Server.discover`                                                                                        |
| Server answers `events/*` with -32601 (HTTP 404) when it did not advertise                                                                                                                                 | Capability Negotiation, Fallback                                       | `Server._handle`                                                                                         |
| Client MUST NOT send `events/*` to a server that did not advertise                                                                                                                                         | Capability Negotiation, Fallback                                       | `Server._handle` (violation `client-must-not-send-events-without-capability`)                            |
| Results carry `resultType: "complete"`                                                                                                                                                                     | Specification, conventions                                             | every handler                                                                                            |
| `delivery` is a non-empty subset of poll/push/webhook; client uses an offered mode                                                                                                                         | Listing Available Events                                               | `Server.poll/stream/subscribe` (violation `client-must-use-offered-delivery-mode`, `Unsupported` -32026) |
| `notifications/events/list_changed` only on `subscriptions/listen`, only if opted in; `eventsListChanged` echoed in acknowledged only if honoured; acknowledged first                                      | Dynamic event types; 2026-07-28 Subscriptions                          | `Server._listen`                                                                                         |
| Purely additive changes do not terminate; removal is `NotFound{kind:event}`; in-place change is `Unsupported{feature,reason:schema_changed}`, never `Forbidden`                                            | Event type removal and breaking changes                                | control `remove_type`/`schema_change`                                                                    |
| Poll `cursor:null` means "now": no events, a fresh cursor; `hasMore`, `maxEvents`, `nextPollMs`; truncation is a result, never an error                                                                    | Poll-Based Delivery; Cursor Lifecycle                                  | `Server.poll`                                                                                            |
| Poll errors are JSON-RPC errors                                                                                                                                                                            | Poll-Based Delivery                                                    | `Server.poll`                                                                                            |
| Push: invalid subscription answers a JSON-RPC error and opens no stream; `notifications/events/active` first; every message carries the request id as `subscriptionId`                                     | Push-Based Delivery, Event delivery                                    | `Server.stream`, `_stream`                                                                               |
| Push: periodic heartbeat with the cursor; `heartbeatMs` granted and authoritative                                                                                                                          | Lifecycle                                                              | `Server._stream`                                                                                         |
| Push: cancellation (HTTP stream close, stdio `notifications/cancelled`) stops delivery and releases resources                                                                                              | Lifecycle, Cancellation                                                | `_stream` `finally`, `run_stdio`                                                                         |
| Push: terminated notification then the empty final result                                                                                                                                                  | Subscription Termination                                               | `Server._stream`                                                                                         |
| `events/subscribe`: `Forbidden` without an authenticated principal                                                                                                                                         | Subscription identity                                                  | `Server.subscribe`                                                                                       |
| Client: `delivery.secret` is `whsec_` + base64 of 24 to 64 bytes; rejected with `InvalidParams`                                                                                                            | Webhook security, Secret generation                                    | `Server.subscribe` (violation `client-secret-is-whsec-24-to-64-bytes`)                                   |
| Client: callback `https`; server rejects other schemes `InvalidParams` (unless `--allow-http-callback`)                                                                                                    | Webhook security, TLS requirement                                      | `Server.check_url` (violation `client-callback-url-https`)                                               |
| Arguments valid for `inputSchema`, else `InvalidParams`                                                                                                                                                    | Error Handling                                                         | `Server.validate_args` (violation)                                                                       |
| Server: derived deterministic `id`, in `X-MCP-Subscription-Id` on every POST                                                                                                                               | Subscription identity; Webhook event delivery                          | `Server.subscribe`, `Server.post`                                                                        |
| Server: `refreshBefore` is the grant, never `null` unless the client asked; client MUST refresh before it                                                                                                  | Subscription TTL; Subscribing                                          | `Server.subscribe` (violation `client-must-refresh-before-refreshBefore`, 0.75 s slack)                  |
| Server: no delivery before the verification handshake; constant-time compare; failure is `CallbackEndpointError` with the category                                                                         | Endpoint verification                                                  | `Server.verify`                                                                                          |
| Server: Standard Webhooks `webhook-id`/`-timestamp`/`-signature` on every delivery, `v1,` HMAC over the raw body, timestamp and signature regenerated per attempt, `msg_<type>_<random>` ids for envelopes | Signature scheme                                                       | `Server.sign`, `Server.post`                                                                             |
| Server: dual-sign on secret rotation                                                                                                                                                                       | Secret rotation                                                        | `Server.subscribe`, `Server.sign`                                                                        |
| Server: `v1a,` Ed25519, key on the Server Card at the MCP origin as a JWK set with `kid`                                                                                                                   | Server identity                                                        | `Server.sign`, `Server.card`                                                                             |
| Server: no redirects; `410` and `413` non-retryable; retries bounded                                                                                                                                       | Webhook event delivery, Delivery profile                               | `Server.http`, `Server.pump`                                                                             |
| Server: `cursor` in a webhook payload is a safe watermark; absent means `null`                                                                                                                             | Webhook event delivery; Cursor Lifecycle                               | `Server.pump`, `_ev`                                                                                     |
| Client: MUST NOT persist or replay from a `null` cursor                                                                                                                                                    | Cursor Lifecycle                                                       | `start_position` (violation `client-must-not-replay-from-null-cursor`)                                   |
| Server MUST NOT send notification types the client has not explicitly requested; acknowledged reflects the subset honoured; tools/resources `listChanged` advertised                                       | 2026-07-28 Subscriptions                                               | `Server._listen`, `Server.discover`                                                                      |
| Server: `terminated` envelope, signed, on revoke/removal/schema change; subscription no longer exists                                                                                                      | Subscription Termination                                               | `Server.terminate_subs`                                                                                  |
| `events/unsubscribe` resolves by `(principal, url, name, arguments)`; unknown is `NotFound{kind:subscription}`                                                                                             | Unsubscribing                                                          | `Server.unsubscribe`                                                                                     |
| Error codes -32023 .. -32027 and `data` shapes                                                                                                                                                             | Error Handling                                                         | throughout                                                                                               |
| Client MUST set `Mcp-Name` to `params.name` on poll/stream/subscribe/unsubscribe; Base64 sentinel when not plain ASCII                                                                                     | Routing Headers                                                        | `Server.validate_headers` (400, -32020, violation)                                                       |
| `events/list` carries no `Mcp-Name`; `Mcp-Method` is the method                                                                                                                                            | Routing Headers                                                        | `Server.validate_headers` (violation `events-list-has-no-mcp-name`)                                      |
| Client MUST set `Mcp-Param-{Name}` for each annotated property that has a value; encoded; server validates and rejects a mismatch with 400 + -32020                                                        | Custom headers from subscription arguments; 2026-07-28 Streamable HTTP | `Server.validate_headers`                                                                                |
| `MCP-Protocol-Version` header present and equal to `_meta`; unsupported version is -32022                                                                                                                  | 2026-07-28 Protocol Version Header                                     | `Server.validate_headers`, `Server._handle`                                                              |
| `Origin` validated (403); GET/DELETE on the endpoint 405; notifications 202; `Accept` lists both types                                                                                                     | 2026-07-28 Streamable HTTP                                             | `make_app`, `validate_headers`                                                                           |
| Client MUST NOT send JSON-RPC responses                                                                                                                                                                    | 2026-07-28 Sending Messages                                            | `make_app`, `run_stdio`                                                                                  |

### Not covered

- SSRF at delivery time (resolve, check the IP, connect to the validated IP). The server validates at
  subscribe time and does not pin: the behaviour is the server's, and nothing in the client's conduct can
  fail it.
- A stream that the client abandons mid-heartbeat on a 3x-heartbeat watchdog (client SHOULD).
- `truncated: true` with a gap. The server implements it (`start_position`, the `maxAgeMs` floor) and
  `selftest.py` does not yet drive a real gap.

## Ambiguities noticed

- Webhook with an authenticated principal over stdio: the SEP requires "an authenticated principal" for
  `events/subscribe`; stdio has no authorization layer. The server treats the process as the principal.
- "A client that receives no message for three times the granted `heartbeatMs`" is a SHOULD with no
  prescribed action on stdio ("a server MAY grant a longer interval").
- Whether a client may resend a pre-change cursor to `events/stream` / `events/subscribe` after a
  `schema_changed` termination: the SEP specifies epoch cursors only for poll, so the server accepts old
  cursors there and answers `Unsupported` only to poll.
