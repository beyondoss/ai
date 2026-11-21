# MCP OAuth interop check

Drives the real `agent` binary's MCP OAuth client (`mcp-login`, then `run`) against an
independent, official implementation: the Python MCP SDK's `examples/servers/simple-auth`, a
separate authorization server (AS) and a protected MCP resource server (RS).

Pinned: `mcp==2.3.0` (PyPI) and the example at tag `v2.3.0`, sha
`2118f14f8a19bc158d8a1cf90af58d85d187f849` (`requirements.txt`).

```sh
uv venv venv && VIRTUAL_ENV=venv uv pip install -r crates/agent/eval/mcp_oauth_interop/requirements.txt
cargo build -p beyond-agent --bin agent          # once per tree you want to check
AGENT_BINS=main=/path/to/main/agent,pr=/path/to/head/agent CHECK_LOG_DIR=./logs \
  venv/bin/python crates/agent/eval/mcp_oauth_interop/check.py
```

`AGENT_BINS` is `label=path,...`. The label `main` is the baseline before tokens were bound to
their server (no URL binding, no PKCE refusal); any other label is expected to behave as the
binding change does. `A4` (legacy entry) needs a `main` binary (or `AGENT_LEGACY_BIN`) to write the
old-format credential. Exit status is nonzero if any observation differs from what that label is
expected to do. It uses a throwaway `HOME` per case, so `~/.beyond` is never read or written.

## How it runs, and the browser step

- AS and RS run unmodified (RS with `--oauth-strict`, so tokens must be issued for its own URL).
- No browser is needed. The example's login page is a plain HTML form with its demo credentials
  printed on it, so the check follows the printed authorization URL over HTTP, submits the form
  (`demo_user` / `demo_password`) and delivers the redirect to the agent's loopback callback, which
  is what a browser would do.
- `agent run` is given a scripted model (`fake_model.py`, Anthropic SSE) that calls the RS's
  `get_time` tool once, so the tool call goes through the real MCP client path.
- Two assertions need an AS the example does not ship, so `as_variant.py` varies it **in its own
  process only** (the SDK is not edited): `AS_VARIANT=no-pkce` drops `code_challenge_methods_supported`
  from the metadata; `AS_VARIANT=refresh` replaces the example's "refresh tokens not supported"
  stubs with a working `refresh_token` grant and a 3 s access-token life.
- The "other server" B is `lookalike.py`: a bare listener that logs, per request, only whether an
  `Authorization` header was present. Credential values are never logged; logs are scrubbed.

## Assertions

| id | what                                                                                              | expected main          | expected binding change                                                                                     |
| -- | ------------------------------------------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| A1 | `mcp-login` succeeds, a `run` calls a tool with the token; the stored entry records the URL (A1c) | works; no URL recorded | works; URL recorded                                                                                         |
| A2 | login meets the RS's 401 challenge and fetches PRM from its `resource_metadata` URL               | yes                    | yes                                                                                                         |
| A3 | same server name re-pointed at another URL B: is the token sent to B?                             | sent                   | not sent; refusal names `mcp-login` (specific text with `RUST_LOG=warn`); back at the original URL it works |
| A4 | credential written by `main`'s binary used by the new binary                                      | n/a                    | refused (re-login hint); re-login then works                                                                |
| A5 | AS metadata omits `code_challenge_methods_supported`                                              | login proceeds         | login refused, names PKCE                                                                                   |
| A6 | access token expired (3 s), refresh token held                                                    | refreshes              | refreshes (bound credential)                                                                                |

simple-auth advertises PRM both ways (the 401 `WWW-Authenticate` challenge's `resource_metadata`
and the well-known URI, at the same path-suffixed URL), so A2 cannot tell the two discovery paths
apart: a client that only tries the well-known URI passes it too. The hermetic fixture tests
(`tests/mcp_oauth_binding.rs`) cover a challenge pointing somewhere else.

## CI

Weekly in `deep.yml` (job `mcp-oauth-interop`), not on PRs: it builds the agent and installs the SDK
from GitHub, a few minutes, no browser. It runs the checked-out tree's binary only (label `head`),
so it asserts the post-binding behaviour and skips A4.

## Client credentials and step-up against Keycloak (`check_cc.py`)

The other half of this kit. `check_cc.py` drives the agent's OAuth **client credentials** (the
`io.modelcontextprotocol/oauth-client-credentials` extension) and **step-up on 403 `insufficient_scope`**
against an independent authorization server: **Keycloak 26.4**, pinned
`quay.io/keycloak/keycloak:26.4@sha256:9409c59bdfb65dbffa20b11e6f18b8abb9281d480c7ca402f51ed3d5977e6007`,
in Docker (containers are named `aimcpcc-<pid>-kc` and removed on exit). Needs Docker and the venv above.

```sh
AGENT_BINS=main=/path/to/main/agent,pr=/path/to/head/agent CHECK_LOG_DIR=./logs \
  venv/bin/python crates/agent/eval/mcp_oauth_interop/check_cc.py
```

The protected server is `rs_server.py`: the official Python MCP SDK's `MCPServer` in resource-server mode
(`token_verifier`, `AuthSettings`, its RFC 9728 metadata), adapted from the `simple-auth` example's
resource server with exactly these changes: a JWT/JWKS token verifier instead of the demo AS's
introspection, an ASGI gate that answers a `tools/call` lacking the tool's scope with `403` +
`WWW-Authenticate: Bearer error="insufficient_scope", scope="..."` (the SDK enforces one scope for the
whole endpoint only), two tools (`get_time` needs `mcp:read`, `write_note` needs `mcp:write`), and a JSON-lines
request log (method, tool, whether the extension was declared, whether `Authorization` was present,
the token's scopes and `jti`; never a token value).

Keycloak's realm (generated per run): scopes `mcp:read`, `mcp:write` (optional) and `mcp:admin`
(no client may have it); clients `agent-cc-secret` (secret), `agent-cc-jwt` (`private_key_jwt`, an RSA key
generated per run), `agent-cc-narrow` (only `mcp:read`), `agent-cc-short` (4 s access tokens); an audience
mapper so every token is for the RS; a user and dynamic client registration for the interactive case.

| id   | what                                                                                          | main | with the feature |
| ---- | --------------------------------------------------------------------------------------------- | ---- | ---------------- |
| CC1  | client id + secret from an env var, no browser, no `mcp-login`: `run` calls a protected tool  | no   | yes              |
| CC1b | the extension is declared in `_meta` on every `tools/call`                                    | n/a  | yes              |
| CC1d | an inline literal secret is refused and never echoed                                          | n/a  | yes              |
| CC2  | the same with `private_key_jwt`                                                               | no   | yes              |
| CC3  | after a 4 s token expires the next call re-acquires (a different `jti`)                       | no   | yes              |
| CC4  | the working token is not sent to another URL under the same name                              | no   | yes              |
| SU1  | a tool needing an extra scope: 403, the agent steps up, the call succeeds                     | no   | yes              |
| SU3  | the AS refuses the scope: one 403, a clear error, no loop                                     | no   | yes              |
| SU2  | `mcp-login` server: the error names the scope and `mcp-login kc --scope`; running it fixes it | no   | yes              |

SU2a (`mcp-login` against Keycloak at all) also fails on `main`: Keycloak sends the RFC 9207 `iss` in
the redirect and `main` did not pass it to rmcp's issuer check. SM1 (service mode never activates either
feature) is hermetic: `tools::mcp::tests::client_credentials_never_activate_for_a_grant_connector`.

Keycloak sets `Secure` cookies even on its dev http listener, so the headless login returns them by hand.
It runs in the weekly `deep.yml` job `mcp-oauth-interop` (after the simple-auth check).
