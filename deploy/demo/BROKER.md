# The broker contract: an agent that acts as a signed-in user

This is the contract the demo's stand-in broker (`crates/demo/src/oidc/`) implements, written as the
behaviour a managed-connector endpoint should have when the third party is an OAuth / OIDC service and
the connector acts _as a user_. It is the input to a spec, not the spec: where the demo took a shortcut,
the shortcut is named.

The agent never holds anything but a `bmc_v1` token (`crates/agent/src/managed_mcp_token.rs`, unchanged).
Everything below is behind the managed endpoint.

## Parties and names

| Name              | What it is                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| **tenant**        | the customer; the unit that owns keys. Its id is in every `bmc_v1` token.                                     |
| **user**          | an end user of the tenant, known to the identity provider by `sub`.                                           |
| **connector**     | a managed MCP server in a grant (`notes` here). The broker knows its upstream and the scopes it asks for.     |
| **connection**    | one user's consent to one connector, for one tenant: `(tenant, connector, sub)`. Holds the refresh token.     |
| **session**       | an agent session, named in its token. Bound to at most one connection per connector.                          |
| **binding**       | `(tenant, session, connector) -> connection`, made by the control plane. The only way from a token to a user. |
| **control plane** | whatever starts consents, binds sessions and revokes. Authenticated by an admin credential.                   |

## What an access token is

The upstream is a resource server that accepts only `Authorization: Bearer <JWT>`: signature checked
against the provider's JWKS (RS256 / ES256, nothing else), `iss`, `aud` (`notes-api`), `exp`, and a scope
per tool. The broker sends exactly that and nothing else upstream. It never forwards the `bmc_v1` token,
and the upstream never sees the session, the tenant or the agent.

## Flows

### 1. Consent: once per user and connection

1. The control plane calls `POST /connect/start {tenant, connector, scopes?}`. The broker makes a PKCE
   verifier (48 random bytes), a `state` and a `nonce`, keeps them in memory for 10 minutes, and returns
   the provider's authorization URL: `response_type=code`, `client_id`, `redirect_uri`,
   `scope=openid offline_access <connector scopes>`, `state`, `nonce`, `code_challenge` (S256).
2. The user's browser goes there, signs in and approves the consent screen at the provider. The broker
   never sees the password.
3. The provider redirects to `GET /oauth/callback?code&state`. The broker looks up and **removes** the
   `state` (one time; unknown, expired or reused is refused), exchanges the code with the verifier and its
   own client credentials, and validates the ID token (signature, issuer, audience = its client id, nonce).
   `sub` comes from the ID token. A response without a refresh token is refused (offline access was not
   granted).
4. The broker stores the connection (below). Signing in again as the same user for the same connector and
   tenant renews the same connection (the id is stable).
5. The scope on the connection is what the provider reports it granted, not what was asked.

A user who declines gets `error=access_denied` at the callback: nothing is stored.

### 2. Binding

`POST /sessions/bind {tenant, session, connector, connection}` (control plane). Refused when the
connection is unknown (404), belongs to another tenant or connector (403), is not active, or the session
is already bound to a different connection (409). Binding the same pair again is a no-op. The binding is
persisted. A session has **exactly one** connection per connector, for its life.

### 3. A call

`POST /managed/<connector>/mcp` with `Authorization: Bearer bmc_v1…`:

1. Verify the token (MAC, expiry) and that it names this connector. Failure is `401`.
2. Look up the binding for `(tenant, session, connector)` from the **token's** claims. None: the call is
   answered, not failed (see "Refusals").
3. A request that names a connection (header `x-beyond-connection` or `?connection=`) which is not the bound
   one is refused with `403`. Nothing in a request can choose a connection.
4. Get an access token for the connection: the cached one if it has more than 10 s left, otherwise refresh
   (below). Per-connection mutual exclusion, so concurrent calls make one refresh.
5. Forward the JSON-RPC body with `Authorization: Bearer <access token>`. If the upstream answers `401`
   (the provider ended the token early), force one refresh and retry once.
6. Return the upstream's status and body. `tools/list` results are cached per connector (see "Refusals").

### 4. Refresh

Open the sealed refresh token, `grant_type=refresh_token` at the provider, replace the cached access token
(memory only). If the provider rotated the refresh token, seal and store the new one. `invalid_grant`
means the provider no longer honours the connection: it becomes `reauth_required` and the stored token is
erased. Any other failure is a `502` and changes nothing. The access token's lifetime is the provider's
(60 s in the demo); the agent can neither see nor influence it.

### 5. Revocation

`POST /connections/<id>/revoke` (the user's act, relayed by the control plane):

1. RFC 7009 revocation of the refresh token at the provider.
2. Because a provider answers 200 whether or not it knew the token, the broker **uses** the revoked token
   once; `invalid_grant` is the proof it is dead, and is reported.
3. Erase the sealed token and the cached access token; status `revoked`. The record stays (history), the
   secret does not. Revoking twice is fine.

Access tokens already issued live out their lifetime at the provider; the broker stops using them at once.

## What is stored, and how

Per connection, one file `<id>.json` (written whole or not at all: a temporary file, then a rename):
`id`, `tenant`, `connector`, `sub`, `username`, `scope` (granted), `created_ms`, `status`
(`active` | `revoked` | `reauth_required`), `refreshes`, and `refresh_sealed`. Bindings are one more file
of `tenant,session,connector -> connection`.

`refresh_sealed` is the refresh token under **XChaCha20-Poly1305**:

- key: `HKDF-SHA256(salt = none, ikm = master, info = "broker tenant key v1" ‖ tenant)`, so each tenant has
  its own key, derived from one master that is in the process environment and never on disk;
- nonce: 24 random bytes per seal, stored with the ciphertext (base64url);
- associated data: the connection id, so a sealed token cannot be moved to another connection's record.

A token sealed for one tenant does not open under another tenant's key; the broker's audit endpoint
(`GET /connections/<id>?audit=1`) shows this on the live record. Access tokens are never written anywhere.
Pending consents (verifier, state, nonce) are memory only.

Demo shortcut: the master key is one SSM parameter. A real deployment wants it in a KMS, with the tenant key
a data key wrapped per tenant, and rotation (re-seal under a new key, keep the old one to open).

## Refusals the model can read

A refusal must leave the session alive and say why. For a tool call the broker answers `200` with a tool
result `isError: true` and a plain sentence:

- no binding: "No user has connected their account to this session, so there is nothing to act as…"
- connection revoked or dead: "The connection to this account is no longer active: … The user has to sign
  in again to reconnect it; nothing else is wrong and the session is fine."

`initialize`, `ping`, notifications and `tools/list` are still answered for such a connection (the last
`tools/list` the connector gave is cached), so the connector does not vanish from the session. Scope
refusals are the upstream's, passed through unchanged (`insufficient_scope: … notes:write …`): the broker
does not interpret scopes.

## Token binding, in one paragraph

A `bmc_v1` token proves **which session** is calling (tenant, session, connector, expiry, MAC). It names no
user. The user comes from the binding the control plane made for that session, and a request cannot name
another. Two users in one tenant are therefore isolated by the bindings alone: a stolen or replayed session
token reaches that session's user and no one else; a token for a session nobody connected reaches nothing.

## Logging and secrets

One JSON line per event on stdout (`consent_started`, `connection_created`, `session_bound`,
`managed_call`, `token_refreshed`, `refresh_rejected`, `connection_revoked`, `managed_refused`). No line
holds a token, a code or a secret. For the memory scan the broker keeps the **last 24 characters** of every
token it handles (`GET /connections/<id>/fingerprints`): enough to recognise a token in a dump, not to use it.

## Demo shortcuts, collected

- The provider is Keycloak in dev mode on plain HTTP inside the VPC (a real broker requires HTTPS to the
  provider); the issuer is a stable internal name and the scripted browser reaches it by address.
- Discovery is read once at start; keys (JWKS) are cached 5 min and refetched on an unknown `kid`.
- Connection records are on the container's disk, so they live as long as the task; the code reloads them
  and the bindings at start.
- One client secret for one connector; a real broker has one registered client per connector and tenant
  policy about which scopes a connector may ask for.
- The control plane is a bearer on a port open to the caller's address.
