# What Beyond must provide for hosted agents

This repo owns the agent and the contracts it speaks; Beyond owns the control plane, the edge and the
sandboxes behind them. This document is the durable list of what Beyond's side must do, and it must be
read when Beyond implements its part. It sits beside [FLEET.md](FLEET.md) (the edge contract and
operations) and [ARCHITECTURE.md](ARCHITECTURE.md) (the mechanism).

- [Must fix before hosted agents go live](#must-fix-before-hosted-agents-go-live)
- [Principles](#principles)
- [Decisions](#decisions)
- [Gateway credentials: expiry and signing keys](#gateway-credentials-expiry-and-signing-keys)
- [What Beyond builds, per part](#what-beyond-builds-per-part)
- [Conformance](#conformance)

## Must fix before hosted agents go live

**Narrow who may write `edge-agent-keys`.** Today every Beyond service (API, orchestrator, edge,
instd) shares one NGS credential, so any of them can write the `edge-agent-keys` bucket, which means
any of them can publish agent-token verification keys. Beyond's own assessment is that the stakes are
bounded: a forged edge key only gets a request past the edge, because `sandbox-exec` still checks the
key sealed into the VM. What it buys is waking sandboxes.

That still matters for hosted agents. They multiply the number of sandboxes behind agent-token routes,
so forged-key wakes become a cost and availability attack on every tenant. Narrow this first.

The requirement: a dedicated API NGS user allowed to publish `$KV.edge-agent-keys.>`, and a deny on
that subject for the shared user. A KV put is a publish to that subject, and NATS can scope it only
per user. A separate bucket makes it one subject rather than a prefix inside the routing bucket.

**Acceptance check:** a service other than the API publishing to `$KV.edge-agent-keys.>` is refused by
NATS.

Source: Beyond's `edge/ARCHITECTURE.md`, the "Who may write `edge-agent-keys`" bullet. It states the
shared credential and the narrowest fix above.

## Principles

1. **The agent never holds a long-lived secret.** Everything it holds arrives in a grant, is scoped to
   one session and expires.
2. **An agent is a deployment.** An immutable bundle named by its SHA-256, the way an image is named by
   its digest. Nothing falls back to another definition.
3. **No sandbox until a tool needs one.** Opening a session, reading the definition and listing tools
   create no VM.
4. **Everything is bounded, with a stated cap and an owner.** A number nobody owns is a number nobody
   enforces.
5. **The contract lives in this repo.** Wire formats, golden vectors and a conformance kit are here;
   Beyond implements against them and never the other way round.

## Decisions

All agreed 2026-10-24.

1. **Room key.** Reuse Beyond's existing agent token (`bagt_v1`, `rustlib/agent-token`): Ed25519 by
   kid, minted by the API, scoped to one sandbox, valid as long as the sandbox lives. The edge checks it
   on every path, health checks included, and `sandbox-exec` checks it again inside the VM. The grant's
   exec headers carry it. On E2B the agent still calls the one Beyond address: the edge checks the token
   and forwards using Beyond's own E2B credentials, which the agent never sees, and `sandbox-exec` in the
   E2B VM checks the same token.
   _Why:_ one verifier already exists in three places and is pinned to its minter by a fixture, and the
   VM check means a bug at the edge is not a breach.
2. **Token lifetimes.** The grant is a ticket to open: valid for minutes, re-minted on resume. The
   tokens inside it live for the session, with a hard ceiling around 24 h, and are revoked when it ends.
   _Why:_ a leaked grant is useless within minutes; a leaked token is useless once the session is gone.
3. **`bmc_v1`.** It follows decision 2. The broker refreshes upstream user tokens itself, and the agent
   never sees a user token.
   _Why:_ a replica compromise then yields a session-scoped token, not a user's third-party credential.
4. **Sandbox affinity.** The exec front picks a chat's home once, at its first sandbox call: Beyond
   normally, E2B only when Beyond is full, BYOC if configured, within the tenant policy carried in the
   token. That choice never changes for the chat's life, and nothing migrates. Beyond homes are durable
   (GlideFS). E2B homes survive idle by pause and resume, and a lost E2B VM loses that chat's files;
   this is documented and accepted.
   _Why:_ migration is the expensive, failure-prone part, and a stable home keeps files and processes
   where the model left them.
5. **Warm snapshots.** One per base stack (coding, browser), built from public images only, with the
   bundle seeded after resume. This extends Beyond's derived-snapshot cache
   (`compute/docs/02-instance-lifecycle.md`).
   - MUST reseed kernel randomness and regenerate machine ids on every resume, because resuming one
     memory image clones its RNG state across tenants.
   - MUST take the snapshot before any key, TLS session or random token exists.
   - MUST never include tenant data.

   _Why:_ it makes the first sandbox call cheap without letting one tenant's VM share secrets or
   randomness with another's.

6. **Bundle retention.** Keep every hash a non-deleted session pins, plus the last 10 deploys per
   agent. Delete all of a tenant's bundles when the tenant is deleted.
   _Why:_ a session must always be able to reopen the definition it started with, and nothing else
   needs to outlive its tenant.
7. **Transport.** TLS on every hop, tokens scoped and never logged, no mTLS from the replica.
   _Why:_
   - TLS stops eavesdroppers.
   - A compromised replica would hold a client certificate too, so mTLS from it proves nothing the
     token does not.
   - Logs are handled by never logging tokens.
   - mTLS already exists inside Beyond between the edge and the instd gateways
     (`edge/ARCHITECTURE.md`, "mTLS Gateway Routing").

## Gateway credentials: expiry and signing keys

The gateway verifies a managed key (`bai_v1`, `bai_v2`, `bai_v3`) with no lookup, from a signature
under a public key it trusts. Layouts, the leeway and the contract vectors are in
`crates/gateway/ARCHITECTURE.md` (Identity) and `verify/key_vectors.json`. This is what Beyond's
control plane must provide for the short-lived credentials and live key rotation they allow. It
states the contract, not how Beyond meets it.

1. **Mint `bai_v3` for short-lived credentials.** A `bai_v3` key is a `bai_v2` key plus a mandatory
   `exp` (unix seconds). The gateway refuses it once `now > exp + 30s` with a 401, `key expired`.
   Expiry is judged when a request is admitted; a stream already admitted is not cut. Anything that
   should not expire stays `bai_v2`. The minter must reproduce `verify/key_vectors.json` byte for byte.
2. **Exchange a workload identity for a `bai_v3`.** An RFC 8693 style token exchange. A workload
   presents the OIDC token its platform issued (AWS, GCP, GitHub Actions, or our hosted agents).
   Beyond checks the token's signature against the issuer's keys and its `iss`, `aud` and `sub`
   against the tenant's trust rules, and returns a `bai_v3` for that tenant with `exp` about an hour
   out. A token that matches no rule gets nothing; a rule can be removed, and no key minted after that
   is issued under it. The workload asks again before `exp`; the gateway never refreshes anything.
3. **Publish and retire signing keys through KV.** To rotate: write `signing_key.{kid}` to the KV
   bucket the gateway watches (value: the public key, base64 or raw, as `signing_keys` takes it)
   and mint under that `kid` once replicas have it. The gateway holds at most 64 such keys, so delete
   `signing_key.{kid}` for a retired kid. To cut a kid off at once, from any source, write
   `blackhole.kid.{kid}`; it beats both config and KV and is checked before any signature math.
   Deleting that entry restores the kid. Writes to these two prefixes decide who can mint for any
   tenant, so the principal allowed to write them is the one that may hold a signing key.
4. **Stop minting `bai_v1`.** It names tenant and VPC only, so no single credential can be cut off
   without cutting off the tenant. New keys are `bai_v2` (long-lived) or `bai_v3` (short-lived).

Long-lived `bai_v2` keys remain the SDK path: an SDK sends a static string and never refreshes, so
only a workload that can run the exchange gets a `bai_v3`.

**Acceptance check:** a `bai_v3` presented after `exp` plus 30s is a 401 `key expired`; a kid
written to `blackhole.kid.{kid}` refuses every token under it, whichever source holds its key; a
token under a kid written to `signing_key.{kid}` is admitted with no restart and refused once the entry
is deleted.

## What Beyond builds, per part

| Part                   | Must do                                                                                                                                                                                                                                                                                                            | Acceptance check                                                                                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Control plane: grants  | Mint `bsg_v1` grants (`src/grant.rs`) that are valid for minutes and re-minted on resume. Carry the tenant policy, `limits` and exec headers. Revoke the session's tokens when it ends.                                                                                                                            | A grant older than its expiry opens nothing. A session's token is refused after the session ends.                                                                                                       |
| Control plane: bundles | Build a plain `ustar` bundle at deploy (`src/bundle.rs`) and refuse unsafe entries there: absolute or `..` paths, symlinks, hardlinks, devices, pax or GNU headers, and anything over the size cap. Store it content-addressed. Save `.mcp-tools.json` at its root for in-sandbox MCP servers (`src/mcp_json.rs`). | A bundle with an unsafe entry fails the deploy, not the session open. A stored bundle hashes to its name. A session over a bundle with `.mcp-tools.json` opens with no exec request and no MCP request. |
| Exec front / edge      | Make the VM on the first request. One create per session. Kill a VM whose create failed. Cap VMs per tenant and answer `503` when full. Time out idle clients. No entry outlives its VM. Follow the exec protocol (`src/exec_endpoint.rs`, exec v1.4) and the edge contract in [FLEET.md](FLEET.md).               | N concurrent first requests for one session create one VM. A failed create leaves no VM. The tenant's N+1th VM gets `503`. A VM that is gone leaves no route entry.                                     |
| Connector broker       | Authorization code with PKCE. Refresh tokens encrypted per tenant. Mint `bmc_v1` (`src/managed_mcp_token.rs`) bound to one user's connection, with a short expiry. Support revocation and scope.                                                                                                                   | The broker reproduces `tests/fixtures/managed_mcp/v1.json` byte for byte. A token for one connection cannot reach another's. A revoked connection is refused at once. An expired token is refused.      |
| Control plane: keys    | Mint `bai_v3` with `exp`; exchange workload OIDC tokens for it; publish `signing_key.{kid}` and `blackhole.kid.{kid}`; stop minting `bai_v1`. See the section above.                                                                                                                                               | An expired `bai_v3` is refused. A rotated-in kid is admitted without a restart. A distrusted kid is refused.                                                                                            |
| Bundle store           | Serve bundles by hash over TLS. Retain per decision 6.                                                                                                                                                                                                                                                             | A pinned hash is still served after 10 newer deploys. A deleted tenant's bundles are gone.                                                                                                              |

## Conformance

A conformance kit in this repo will check each part against these contracts. Until it lands, the
acceptance checks above are the bar.
