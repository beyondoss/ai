"""The official simple-auth authorization server, optionally varied for two assertions. Nothing
in the SDK checkout is edited: the variation is applied here, at import time, in this process only.

    as_variant.py --port N                  stock behaviour (what the example ships)
    AS_VARIANT=no-pkce  ... --port N        metadata omits `code_challenge_methods_supported`
    AS_VARIANT=refresh  ... --port N        the example's own "refresh tokens not supported" stubs are
                                            replaced by a working refresh_token grant, access tokens
                                            live AS_ACCESS_TTL seconds (default 3)
"""
import os
import secrets
import time

import mcp.server.auth.routes as routes
from mcp.server.auth.provider import AccessToken, RefreshToken
from mcp.shared.auth import OAuthToken
import mcp_simple_auth.auth_server as auth_server

variant = os.environ.get("AS_VARIANT", "")

if variant == "no-pkce":
    _build = routes.build_metadata

    def build_metadata(*a, **kw):
        m = _build(*a, **kw)
        m.code_challenge_methods_supported = None
        return m

    routes.build_metadata = build_metadata

if variant == "refresh":
    TTL = int(os.environ.get("AS_ACCESS_TTL", "3"))
    Base = auth_server.SimpleAuthProvider

    class RefreshProvider(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.refresh_tokens: dict[str, RefreshToken] = {}

        def _issue(self, client_id, scopes, resource, subject) -> OAuthToken:
            access = f"mcp_{secrets.token_hex(32)}"
            refresh = f"mcp_r_{secrets.token_hex(32)}"
            self.tokens[access] = AccessToken(
                token=access, client_id=client_id, scopes=scopes,
                expires_at=int(time.time()) + TTL, resource=resource, subject=subject)
            self.refresh_tokens[refresh] = RefreshToken(
                token=refresh, client_id=client_id, scopes=scopes, resource=resource, subject=subject)
            return OAuthToken(access_token=access, token_type="Bearer", expires_in=TTL,
                              scope=" ".join(scopes), refresh_token=refresh)

        async def exchange_authorization_code(self, client, authorization_code):
            if authorization_code.code not in self.auth_codes:
                raise ValueError("Invalid authorization code")
            del self.auth_codes[authorization_code.code]
            return self._issue(client.client_id, authorization_code.scopes,
                               authorization_code.resource, authorization_code.subject)

        async def load_refresh_token(self, client, refresh_token):
            return self.refresh_tokens.get(refresh_token)

        async def exchange_refresh_token(self, client, refresh_token, scopes):
            del self.refresh_tokens[refresh_token.token]
            return self._issue(client.client_id, scopes or refresh_token.scopes,
                               refresh_token.resource, refresh_token.subject)

    auth_server.SimpleAuthProvider = RefreshProvider

if __name__ == "__main__":
    auth_server.main()
