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

Token epochs (decision 2) raise the stakes. They travel on the same key channel, so a service that
can write it can raise an epoch and revoke a live session's token. The verifiers' raise-only rule
stops it reviving a revoked one, but only the narrowed credential stops the forced revocation.

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

All agreed 2026-10-24. Decisions 1-5 were amended and 8-12 added the same day, after Beyond checked
each one against its code; the reasons are inline.

**Where Beyond's edge sits.** Beyond's edge is on the path from the agent to its sandbox, and only
there. Clients reach the agent cells through the fleet's own front, which owns the callback rules E1-E7
and the ring routing in [FLEET.md](FLEET.md). Beyond's edge does neither.

1. **Room key.** Reuse Beyond's existing agent token (`bagt_v1`, `rustlib/agent-token`): Ed25519 by
   kid, minted by the API, scoped to one sandbox, with the lifetime in decision 2. The edge checks it on
   every sandbox path, sandbox health checks included, and `sandbox-exec` checks it again inside the VM.
   The agent cells' own `/livez` and `/readyz` are the fleet's and need no grant. The grant's exec
   headers carry it, and it is the only bearer on the exec path: there is no second per-sandbox exec
   bearer. On any other provider (E2B, BYOC or anything else) the agent still calls the one Beyond
   address: the edge checks the token and forwards using Beyond's own credentials for that provider,
   which the agent never sees, and `sandbox-exec` in that provider's VM checks the same token.
   `bagt_v1` is Beyond's contract. The agent carries it as an opaque header and never verifies it, so
   its format and vectors live in Beyond's repo, not this one.
   _Why:_ one verifier already exists in three places and is pinned to its minter by a fixture, and the
   VM check means a bug at the edge is not a breach. One bearer per door is less to leak.
2. **Token lifetimes.** The grant is a ticket to open: valid for 5 minutes, re-minted on resume. The
   tokens inside it expire at the session's end or after 24 h, whichever comes first. Revocation is a
   per-sandbox token epoch: each `bagt_v1` carries one, and the edge and `sandbox-exec` refuse any token
   below the sandbox's current epoch, which they learn through the key channel they already watch.
   Ending a session bumps the epoch. The edge and `sandbox-exec` only ever raise a sandbox's epoch
   and never accept a lower one, so nothing that can write the key channel can bring a revoked token
   back.
   _Why:_ a leaked grant is useless within minutes; a leaked token is useless once the session is gone.
   An epoch is one number per sandbox, takes effect at once, and needs no denylist. 5 minutes covers
   clock skew and the open; the grant has no leeway.
3. **`bmc_v1`.** It expires with the same 24 h ceiling, which is never shorter than the grant that
   carries it. It is revoked at the broker: every call checks the session's binding, deleting a session
   removes its bindings, and a revoked connection is refused at once. The broker refreshes upstream user
   tokens itself, and the agent never sees a user token.
   _Why:_ a replica compromise then yields a session-scoped token, not a user's third-party credential,
   and revocation lives where the binding state already is.
4. **Sandbox affinity.** The exec front picks a chat's home once, at its first sandbox call: Beyond
   normally, another provider only when Beyond is full, BYOC if configured, within the tenant's policy.
   The policy is held by Beyond's control plane and looked up by tenant at the exec front; no token
   carries it. One sandbox serves one chat home and never two tenants. That choice never changes for the
   chat's life, and nothing migrates. Beyond homes are durable (GlideFS). Homes on other providers
   survive idle only as far as that provider pauses and resumes, and a lost VM there loses that chat's
   files; this is documented and accepted.
   _Why:_ migration is the expensive, failure-prone part, and a stable home keeps files and processes
   where the model left them. Policy in a token can be stale or tampered with; policy at the front
   cannot.
5. **Warm snapshots.** One per base stack (coding, browser), built from public images only, with the
   bundle seeded after resume. This extends Beyond's derived-snapshot cache
   (`compute/docs/02-instance-lifecycle.md`).
   - MUST reseed kernel randomness on every resume, before any workload code runs, because resuming
     one memory image clones its RNG state across tenants. VMGenID alone leaves a window between the
     vCPUs resuming and the reseed (Firecracker's `random-for-clones.md`), so Beyond closes it with an
     explicit reseed or a virtio-rng device, and proves it live.
   - MUST regenerate everything per-machine that is load-bearing: SSH host keys, MAC, boot id and the
     CSPRNG. `/etc/machine-id` stays shared. Rewriting it in a restored VM broke dbus and `systemctl`,
     and a shared machine-id collides journal directories and derived ids, which is wrong but not
     unsafe. Firecracker's clone guidance does not mention it.
   - MUST take the snapshot before any key, TLS session or random token exists, and with no
     `random-seed` file in the image.
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
8. **Bundle URLs.** Every bundle URL in a grant is https: a short-lived signed link, re-minted with each
   grant, that stays valid for the grant's life plus the replica's 120 s fetch window.
   _Why:_ the hash proves integrity, not confidentiality, and bundles are tenant code.
9. **Exec front.** It gets or creates the sandbox, single-flight on the session id: the first sandbox
   request creates the VM, and every later one goes through the existing wake path. A tenant at its VM
   cap gets `503` with `Retry-After`. A refused extension (stream, job or notify) is `400` or `422`. A
   waking sandbox is never answered with a placeholder `200`, and a create or wake finishes, or streams
   its first frame, inside the agent's 10 s probe and 30 s silence limits.
   _Why:_ these are the statuses the agent's suspect-and-re-probe logic already reads, so the front's
   vocabulary is part of the contract.
10. **Kept jobs and sleep.** A running kept job keeps its sandbox awake until the job ends, and nothing
    of Beyond's contacts a sandbox with unheld jobs on a timer. The agent's 60 s idle reap is not a
    session end and leaves jobs alone. Deleting a session wipes its jobs or destroys its sandbox. Beyond's
    native endpoint advertises notify, because its kept jobs survive sleep; an adapter for another
    provider advertises it only once a live test shows that provider keeps jobs alive while idle (N1).
    _Why:_ a paused VM runs nothing, and the fleet expects a build to finish with no agent attached.
11. **Keys Beyond holds.** Beyond's minter derives the same per-tenant `dek` on every re-mint, stores it
    durably, rotates it by plan and destroys it when the tenant is deleted, which crypto-erases the
    tenant's sessions. Beyond holds the seal key's public half and the managed-MCP secret. The seal key
    also derives the `bjn_v1` notify key, so a replica holding one seal key strands every outstanding
    notify token when it rotates. The agent accepts two seal keys during a rotation (#495), the way it
    already trusts two signing kids. Grants carry no seal kid, so a rotation goes replicas first, then
    Beyond's minter: every replica starts with `--seal-key-previous` set to the old key and unseals
    with either, and only then does the minter switch to the new key. Notify tokens are verified by
    kid and minted under the current key only.
    _Why:_ a re-mint must never strand stored sessions, and rotating a key must never strand kept jobs.
12. **Session ids and the gateway.** Beyond mints session ids as `<home_shard>.<opaque>`, choosing each
    tenant's home shard from the shards the fleet publishes. Beyond's control plane also mints each
    session's managed model key for the grant's `gateway_key`, writes the gateway's deny and allowance
    entries, and bills its usage rows.
    _Why:_ placement stays the fleet's and minting stays Beyond's; the gateway's key, limits and usage
    are one Beyond-owned loop.

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
| Control plane: grants  | Mint `bsg_v1` grants (`src/grant.rs`) that are valid for 5 minutes and re-minted on resume. Carry `limits` and exec headers. Revoke the session's tokens when it ends.                                                                                                                                             | A grant older than its expiry opens nothing. A session's token is refused after the session ends.                                                                                                       |
| Control plane: bundles | Build a plain `ustar` bundle at deploy (`src/bundle.rs`) and refuse unsafe entries there: absolute or `..` paths, symlinks, hardlinks, devices, pax or GNU headers, and anything over the size cap. Store it content-addressed. Save `.mcp-tools.json` at its root for in-sandbox MCP servers (`src/mcp_json.rs`). | A bundle with an unsafe entry fails the deploy, not the session open. A stored bundle hashes to its name. A session over a bundle with `.mcp-tools.json` opens with no exec request and no MCP request. |
| Exec front / edge      | Make the VM on the first request. One create per session. Kill a VM whose create failed. Cap VMs per tenant and answer `503` when full. Time out idle clients. No entry outlives its VM. Follow the exec protocol (`src/exec_endpoint.rs`, exec v1.4) and the edge contract in [FLEET.md](FLEET.md).               | N concurrent first requests for one session create one VM. A failed create leaves no VM. The tenant's N+1th VM gets `503`. A VM that is gone leaves no route entry.                                     |
| Connector broker       | Authorization code with PKCE. Refresh tokens encrypted per tenant. Mint `bmc_v1` (`src/managed_mcp_token.rs`) bound to one user's connection, with a short expiry. Support revocation and scope.                                                                                                                   | The broker reproduces `tests/fixtures/managed_mcp/v1.json` byte for byte. A token for one connection cannot reach another's. A revoked connection is refused at once. An expired token is refused.      |
| Control plane: keys    | Mint `bai_v3` with `exp`; exchange workload OIDC tokens for it; publish `signing_key.{kid}` and `blackhole.kid.{kid}`; stop minting `bai_v1`. See the section above.                                                                                                                                               | An expired `bai_v3` is refused. A rotated-in kid is admitted without a restart. A distrusted kid is refused.                                                                                            |
| Bundle store           | Serve bundles by hash over TLS. Retain per decision 6.                                                                                                                                                                                                                                                             | A pinned hash is still served after 10 newer deploys. A deleted tenant's bundles are gone.                                                                                                              |

## Conformance

A conformance kit in this repo will check each part against these contracts. Until it lands, the
acceptance checks above are the bar. Beyond tracks every requirement this repo places on it, with its
source line and the proof it needs, in the
[Hosted Agents Ledger](https://claude.ai/artifact/9QK7W4NSWQYVeWBoAEyA3e).
