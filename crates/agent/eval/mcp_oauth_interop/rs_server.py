"""A protected MCP resource server built on the OFFICIAL Python MCP SDK (`mcp.server.mcpserver.MCPServer`,
its `token_verifier` / `AuthSettings` resource-server mode and its RFC 9728 protected-resource
metadata), validating tokens issued by Keycloak, and enforcing a scope PER TOOL.

Adapted from the official `examples/servers/simple-auth` resource server (pin: see requirements.txt),
with exactly these changes, because that example validates by introspection against its own demo
authorization server and enforces one scope for the whole endpoint:

  1. `JwtVerifier` replaces `IntrospectionTokenVerifier`: it checks the token's RS256 signature
     against the authorization server's JWKS (PyJWT), its issuer and expiry, and reads `scope` / `aud`.
  2. `ToolScopeGate` (a small ASGI wrapper around the SDK's own app) answers a `tools/call` for a
     tool whose scope the token lacks with `403` and
     `WWW-Authenticate: Bearer error="insufficient_scope", scope="<required>", resource_metadata=...`.
     The SDK enforces `required_scopes` for the whole endpoint only; this is the per-tool part.
  3. Two tools: `get_time` (needs `mcp:read`) and `write_note` (needs `mcp:write`).
  4. Each request's decision is appended to a JSON-lines file: the JSON-RPC method, the tool, whether
     the request declared the client-credentials extension in `_meta`, whether an `Authorization`
     header was present, the token's scopes, and the HTTP status. Token values are never logged.

Usage: rs_server.py PORT ISSUER LOG"""
import json
import sys
import time

import jwt
import uvicorn
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.server import MCPServer
from pydantic import AnyHttpUrl

PORT, ISSUER, LOG = int(sys.argv[1]), sys.argv[2], sys.argv[3]
SERVER_URL = f"http://127.0.0.1:{PORT}/mcp"
TOOL_SCOPES = {"get_time": "mcp:read", "write_note": "mcp:write"}
EXTENSION = "io.modelcontextprotocol/oauth-client-credentials"


class JwtVerifier(TokenVerifier):
    def __init__(self):
        self.jwks = jwt.PyJWKClient(f"{ISSUER}/protocol/openid-connect/certs")

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            key = self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=ISSUER, options={"verify_aud": False})
        except Exception:
            return None
        aud = claims.get("aud")
        audiences = aud if isinstance(aud, list) else [aud] if aud else []
        return AccessToken(
            token=token,
            client_id=claims.get("azp", "unknown"),
            scopes=claims.get("scope", "").split(),
            expires_at=claims.get("exp"),
            resource=audiences[0] if audiences else None,
            subject=claims.get("sub"),
            claims=claims,
        )


verifier = JwtVerifier()
mcp = MCPServer(
    name="Keycloak-protected MCP server",
    token_verifier=verifier,
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(ISSUER),
        required_scopes=["mcp:read"],
        resource_server_url=AnyHttpUrl(SERVER_URL),
        validate_token_resource=True,
    ),
)


@mcp.tool()
async def get_time() -> dict:
    """The server's time (needs scope mcp:read)."""
    return {"current_time": time.strftime("%Y-%m-%dT%H:%M:%S")}


@mcp.tool()
async def write_note() -> dict:
    """Writes a note (needs scope mcp:write)."""
    return {"note": "written"}


def log(**kw):
    with open(LOG, "a") as f:
        f.write(json.dumps(kw) + "\n")


class ToolScopeGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        messages, body = [], b""
        while True:
            msg = await receive()
            messages.append(msg)
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        replay = iter(messages)

        async def replayed():
            return next(replay, None) or await receive()

        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        try:
            rpc = json.loads(body)
        except ValueError:
            rpc = {}
        rpc = rpc if isinstance(rpc, dict) else {}
        params = rpc.get("params") or {}
        meta = params.get("_meta") or {}
        caps = meta.get("io.modelcontextprotocol/clientCapabilities") or {}
        entry = {
            "method": rpc.get("method"),
            "tool": params.get("name") if rpc.get("method") == "tools/call" else None,
            "cc_declared": EXTENSION in (caps.get("extensions") or {}),
            "authorization_present": "authorization" in headers,
        }
        auth = headers.get("authorization", "")
        access = await verifier.verify_token(auth[7:]) if auth.lower().startswith("bearer ") else None
        entry["scopes"] = access.scopes if access else None
        entry["jti"] = access.claims.get("jti") if access and access.claims else None
        required = TOOL_SCOPES.get(entry["tool"]) if entry["tool"] else None
        if access is not None and required and required not in access.scopes:
            challenge = (f'Bearer error="insufficient_scope", scope="{required}", '
                         f'resource_metadata="http://127.0.0.1:{PORT}/.well-known/oauth-protected-resource/mcp"')
            await send({"type": "http.response.start", "status": 403, "headers": [
                (b"content-type", b"application/json"), (b"www-authenticate", challenge.encode()),
                (b"content-length", b"2")]})
            await send({"type": "http.response.body", "body": b"{}"})
            log(**entry, status=403)
            return
        status = {}

        async def snoop(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        await self.app(scope, replayed, snoop)
        log(**entry, status=status.get("code"))


if __name__ == "__main__":
    uvicorn.run(ToolScopeGate(mcp.streamable_http_app()), host="127.0.0.1", port=PORT, log_level="warning")
