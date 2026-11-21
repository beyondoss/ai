#!/usr/bin/env python3
"""OAuth client credentials and step-up for the agent's MCP client, against an INDEPENDENT, off-the-shelf
authorization server (Keycloak, pinned image digest, in Docker) and a protected MCP server built on
the OFFICIAL Python MCP SDK (`rs_server.py`). NOT part of the default CI gate; see README.md.

    AGENT_BINS=main=/path/to/agent-main,pr=/path/to/agent-pr  python check_cc.py

Needs Docker and the venv from requirements.txt. Exit status is nonzero when any observation differs
from what that binary is expected to do (label `main` = before this feature; any other label = with it)."""
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

import check as base  # the #688 kit: process helpers, scrubbing, the scripted model and the lookalike
from check import HERE, LOGDIR, free_port, record, scrub, spawn, wait_port

KC_IMAGE = ("quay.io/keycloak/keycloak:26.4"
            "@sha256:9409c59bdfb65dbffa20b11e6f18b8abb9281d480c7ca402f51ed3d5977e6007")
PREFIX = f"aimcpcc-{os.getpid()}"
containers = []
RS_LOG = os.path.join(LOGDIR, "rs-requests.jsonl")


def rs_entries(start=0):
    if not os.path.exists(RS_LOG):
        return []
    with open(RS_LOG) as f:
        return [json.loads(l) for l in f.read().splitlines()[start:] if l.strip()]


def rs_mark():
    return len(rs_entries())


def calls(entries, tool):
    return [e for e in entries if e["method"] == "tools/call" and e["tool"] == tool]


def scope(name):
    return {"name": name, "protocol": "openid-connect",
            "attributes": {"include.in.token.scope": "true", "display.on.consent.screen": "false"}}


def realm_json(rs_url, jwk_public):
    """The realm Keycloak imports. Scopes mcp:read / mcp:write are optional client scopes, mcp:admin
    exists but no client may use it. Every issued token carries `aud` = the RS (an audience mapper)."""
    aud = {"name": "aud", "protocol": "openid-connect", "protocolMapper": "oidc-audience-mapper",
           "config": {"included.custom.audience": rs_url, "access.token.claim": "true", "id.token.claim": "false"}}
    scopes = [dict(scope("mcp:read"), protocolMappers=[aud]), dict(scope("mcp:write"), protocolMappers=[aud]),
              scope("mcp:admin")]

    def client(cid, optional, **attrs):
        return {"clientId": cid, "enabled": True, "publicClient": False, "serviceAccountsEnabled": True,
                "standardFlowEnabled": False, "directAccessGrantsEnabled": False, "defaultClientScopes": [],
                "optionalClientScopes": optional, "attributes": attrs, **({"secret": SECRETS[cid]} if cid in SECRETS else {})}

    jwt_client = client("agent-cc-jwt", ["mcp:read", "mcp:write"], **{
        "use.jwks.string": "true", "jwks.string": json.dumps({"keys": [jwk_public]}),
        "token.endpoint.auth.signing.alg": "RS256"})
    jwt_client["clientAuthenticatorType"] = "client-jwt"
    return {
        "realm": "mcp", "enabled": True, "sslRequired": "none", "accessTokenLifespan": 300, "clientScopes": scopes,
        "defaultOptionalClientScopes": ["mcp:read", "mcp:write"],
        "clients": [client("agent-cc-secret", ["mcp:read", "mcp:write"]),
                    client("agent-cc-narrow", ["mcp:read"]),
                    client("agent-cc-short", ["mcp:read", "mcp:write"], **{"access.token.lifespan": "4"}),
                    jwt_client],
        "users": [{"username": "demo", "enabled": True, "email": "d@example.test", "emailVerified": True,
                   "firstName": "Demo", "lastName": "User", "realmRoles": ["offline_access"], "credentials": [{"type": "password", "value": "demo"}]}],
        "components": {"org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy": [
            {"name": "Trusted Hosts", "providerId": "trusted-hosts", "subType": "anonymous",
             "config": {"trusted-hosts": ["127.0.0.1", "localhost"], "host-sending-registration-request-must-match": ["false"],
                        "client-uris-must-match": ["true"]}}]},
    }


SECRETS = {cid: secrets.token_urlsafe(24) for cid in ("agent-cc-secret", "agent-cc-narrow", "agent-cc-short")}


def start_keycloak(rs_url, jwk_public):
    port = free_port()
    d = tempfile.mkdtemp(prefix="kc-realm-")
    os.chmod(d, 0o755)
    with open(os.path.join(d, "mcp-realm.json"), "w") as f:
        json.dump(realm_json(rs_url, jwk_public), f)
    os.chmod(os.path.join(d, "mcp-realm.json"), 0o644)
    name = f"{PREFIX}-kc"
    subprocess.run(["docker", "run", "-d", "--name", name, "-p", f"127.0.0.1:{port}:8080", "-v", f"{d}:/opt/keycloak/data/import:ro",
                    "-e", "KC_BOOTSTRAP_ADMIN_USERNAME=admin", "-e", f"KC_BOOTSTRAP_ADMIN_PASSWORD={secrets.token_urlsafe(12)}",
                    KC_IMAGE, "start-dev", "--import-realm", "--http-host=0.0.0.0"], check=True, capture_output=True)
    containers.append(name)
    issuer = f"http://127.0.0.1:{port}/realms/mcp"
    end = time.time() + 180
    while time.time() < end:
        try:
            if httpx.get(f"{issuer}/.well-known/openid-configuration", timeout=2).status_code == 200:
                return port, issuer, name
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("keycloak never became ready: " + subprocess.run(["docker", "logs", "--tail", "30", name], capture_output=True, text=True).stdout)


class Agent:
    def __init__(self, label, binary, tool):
        self.label, self.binary, self.tool = label, binary, tool
        self.model_log = os.path.join(LOGDIR, f"{label}-{tool}-model-results.jsonl")
        self.model_port = free_port()
        spawn([sys.executable, os.path.join(HERE, "fake_model.py"), str(self.model_port), f"mcp__kc__{tool}", self.model_log],
              f"{label}-{tool}-fake-model.log")
        wait_port(self.model_port)
        self.nrun = 0

    def home(self, server):
        h = tempfile.mkdtemp(prefix="agent-home-")
        os.makedirs(os.path.join(h, ".beyond"))
        self.set_server(h, server)
        return h

    @staticmethod
    def set_server(h, server):
        with open(os.path.join(h, ".beyond", "settings.json"), "w") as f:
            json.dump({"mcp_servers": [dict(server, name="kc", transport="http")]}, f)

    def env(self, home, extra=None):
        env = dict(os.environ, HOME=home, **(extra or {}))
        for k in ["AI_POOL_KEY_ANTHROPIC", "DATABASE_URL", "AI_DIRECT", "AI_PROVIDER", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
                  "AI_AGENT_CONFIG_DIR", "AI_AGENT_LIFECYCLE_URL", "AI_BASE_URL", "AI_API_KEY"]:
            env.pop(k, None)
        return env

    def run(self, home, extra_env=None, rust_log=None):
        """One `agent run` whose scripted model calls the tool once. Returns (tool_result|None, output)."""
        before = os.path.getsize(self.model_log) if os.path.exists(self.model_log) else 0
        cwd = tempfile.mkdtemp(prefix="agent-cwd-")
        env = self.env(home, extra_env)
        if rust_log:
            env["RUST_LOG"] = rust_log
        p = subprocess.run([self.binary, "run", "call the tool", "--gateway-url", f"http://127.0.0.1:{self.model_port}",
                            "--key", "bai_v1.test", "--model", "claude-test", "--max-steps", "6", "--no-session-persistence"],
                           env=env, cwd=cwd, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        res = None
        if os.path.exists(self.model_log):
            with open(self.model_log) as f:
                f.seek(before)
                lines = [json.loads(l) for l in f if l.strip()]
            res = lines[-1] if lines else None
        shutil.rmtree(cwd, ignore_errors=True)
        self.nrun += 1
        out = scrub(p.stderr + p.stdout)
        with open(os.path.join(LOGDIR, f"{self.label}-{self.tool}-run-{self.nrun:02d}.log"), "w") as f:
            f.write(f"exit={p.returncode}\ntool_result={json.dumps(res)}\n--- agent stderr+stdout (scrubbed)\n{base.ANSI.sub('', out)}")
        return res, out

    def login(self, home, issuer, extra_args=()):
        """`agent mcp-login kc [args]` with the browser step done headlessly against Keycloak's own login page."""
        errf = tempfile.NamedTemporaryFile("w+", prefix="login-err-", delete=False)
        p = subprocess.Popen([self.binary, "mcp-login", "kc", *extra_args], env=self.env(home), cwd=home,
                             stdout=subprocess.PIPE, stderr=errf, text=True, start_new_session=True, stdin=subprocess.DEVNULL)
        base.procs.append(p)
        auth_url, end = None, time.time() + 30
        while time.time() < end and auth_url is None and p.poll() is None:
            m = re.search(r"(https?://\S+)", open(errf.name).read())
            if m:
                auth_url = m.group(1)
            else:
                time.sleep(0.1)
        if auth_url is None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
            return False, scrub(open(errf.name).read())
        c = httpx.Client(follow_redirects=False, timeout=20)
        r = c.get(auth_url)
        form = re.search(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"', r.text)
        if not form:  # an already-authenticated session skips the form; Keycloak then redirects straight back
            form = None
        if form:
            action = form.group(1).replace("&amp;", "&")
            # Keycloak marks its cookies `Secure` even on its dev http listener, which a cookie jar will
            # not return over http; send them by hand.
            cookie = "; ".join(v.split(";")[0] for v in r.headers.get_list("set-cookie"))
            r = c.post(action, data={"username": "demo", "password": "demo", "credentialId": ""},
                       headers={"cookie": cookie})
        if r.status_code in (301, 302, 303, 307):
            c.get(r.headers["location"])  # the agent's own loopback callback listener
        try:
            out, _ = p.communicate(timeout=40)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
        e = open(errf.name).read()
        os.unlink(errf.name)
        with open(os.path.join(LOGDIR, f"{self.label}-login.log"), "a") as f:
            f.write(f"--- mcp-login kc {' '.join(extra_args)} exit={p.returncode} form_found={bool(form)} "
                    f"last_status={r.status_code} location={r.headers.get('location', '')[:80]}\n{scrub(e + (out or ''))}\n")
        return p.returncode == 0, scrub(e + (out or ""))


def ok_result(res, needle):
    return bool(res) and not res["is_error"] and needle in res["text"]


ISSUER = ""  # Keycloak's issuer, pinned in every client_credentials block (set in main)


def cc(client_id, **kw):
    return {"client_credentials": dict({"issuer": ISSUER, "client_id": client_id, "scopes": ["mcp:read"]}, **kw)}


def main():
    bins = dict(kv.split("=", 1) for kv in os.environ["AGENT_BINS"].split(","))
    for label, b in bins.items():
        assert os.access(b, os.X_OK), f"{label}: {b} is not executable"
    rs_port = free_port()
    rs_url = f"http://127.0.0.1:{rs_port}/mcp"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    jwk = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="agent-cc-jwt-1", use="sig", alg="RS256")
    global ISSUER
    _, issuer, _ = start_keycloak(rs_url, jwk)
    ISSUER = issuer
    spawn([sys.executable, os.path.join(HERE, "rs_server.py"), str(rs_port), issuer, RS_LOG], "rs.log")
    wait_port(rs_port)
    print(f"[stack] keycloak {issuer} (image {KC_IMAGE.split('@')[1][:19]}...), RS {rs_url}", flush=True)
    sec = lambda cid: {"KC_SECRET": SECRETS[cid]}
    jwt_env = {"KC_KEY": pem}

    for label, binary in bins.items():
        new = label != "main"  # any non-baseline binary carries this feature
        time_tool, write_tool = Agent(label, binary, "get_time"), Agent(label, binary, "write_note")

        # CC1: client id + secret from an env var, no browser, no mcp-login.
        s = {"url": rs_url, **cc("agent-cc-secret", client_secret="$KC_SECRET")}
        home = time_tool.home(s)
        m = rs_mark()
        res, out = time_tool.run(home, sec("agent-cc-secret"))
        seen = rs_entries(m)
        record("CC1 client id + secret from an env var: run calls a protected tool", label, ok_result(res, "current_time"), new,
               (res or {}).get("text", out)[:140].replace("\n", " "))
        record("CC1b the request declared the client-credentials extension in _meta, on every request", label,
               (bool(calls(seen, "get_time")) and all(e["cc_declared"] for e in calls(seen, "get_time"))) if new else True, True,
               f"{len(seen)} requests")
        record("CC1c the secret is not in the agent's output or its settings", label,
               SECRETS["agent-cc-secret"] not in out and SECRETS["agent-cc-secret"] not in open(os.path.join(home, ".beyond", "settings.json")).read(), True)
        # CC1d: a literal secret in settings is refused, and never sent anywhere.
        lit = time_tool.home({"url": rs_url, **cc("agent-cc-secret", client_secret=SECRETS["agent-cc-secret"])})
        m = rs_mark()
        res, out = time_tool.run(lit)
        record("CC1d an inline literal secret is refused (the tool is not called, the value not echoed)", label,
               not ok_result(res, "current_time") and SECRETS["agent-cc-secret"] not in out and not [e for e in rs_entries(m) if e["authorization_present"]], True,
               out.strip()[-160:].replace("\n", " "))

        # CC2: private_key_jwt.
        home = time_tool.home({"url": rs_url, **cc("agent-cc-jwt", private_key="$KC_KEY", signing_algorithm="RS256")})
        res, out = time_tool.run(home, jwt_env)
        record("CC2 private_key_jwt (key from an env var): run calls a protected tool", label, ok_result(res, "current_time"), new,
               (res or {}).get("text", out)[:140].replace("\n", " "))
        record("CC2b the private key is not in the output", label, "PRIVATE KEY" not in out, True)

        # CC3: expiry (4 s token) then the next call re-acquires silently.
        home = time_tool.home({"url": rs_url, **cc("agent-cc-short", client_secret="$KC_SECRET")})
        m = rs_mark()
        res1, _ = time_tool.run(home, sec("agent-cc-short"))
        time.sleep(6)
        res2, out2 = time_tool.run(home, sec("agent-cc-short"))
        jtis = {e["jti"] for e in calls(rs_entries(m), "get_time") if e["status"] == 200}
        record("CC3 call after the 4 s token expired succeeds, with a different token, no human", label,
               ok_result(res1, "current_time") and ok_result(res2, "current_time") and len(jtis) == 2, new,
               f"distinct token ids on 200 calls: {len(jtis)}")

        # CC4: the same server name pointed at another URL gets no token (and the AS is not asked).
        home = time_tool.home({"url": rs_url, **cc("agent-cc-secret", client_secret="$KC_SECRET")})
        res, _ = time_tool.run(home, sec("agent-cc-secret"))
        b_log = os.path.join(LOGDIR, f"lookalike-B-{label}.jsonl")
        b_port = free_port()
        spawn([sys.executable, os.path.join(HERE, "lookalike.py"), str(b_port), b_log], f"lookalike-B-{label}.log")
        wait_port(b_port)
        Agent.set_server(home, {"url": f"http://127.0.0.1:{b_port}/mcp", **cc("agent-cc-secret", client_secret="$KC_SECRET")})
        res_b, out_b = time_tool.run(home, sec("agent-cc-secret"))
        blog = [json.loads(l) for l in open(b_log)] if os.path.exists(b_log) else []
        got = [e for e in blog if e["authorization_present"]]
        print(f"[{label}] CC4 B received {len(blog)} requests, {len(got)} with an Authorization header", flush=True)
        record("CC4 the working CC token is not sent to another URL under the same name", label,
               ok_result(res, "current_time") and not got and not ok_result(res_b, "current_time"), new, f"B requests={blog[:3]}")

        # CC5: a server that names an authorization server of its own (the issuer is pinned) receives no credential.
        a_log = os.path.join(LOGDIR, f"attacker-{label}.jsonl")
        a_rs, a_as = free_port(), free_port()
        spawn([sys.executable, os.path.join(HERE, "attacker.py"), str(a_rs), str(a_as), a_log], f"attacker-{label}.log")
        wait_port(a_rs)
        wait_port(a_as)
        home = time_tool.home({"url": f"http://127.0.0.1:{a_rs}/mcp", **cc("agent-cc-secret", client_secret="$KC_SECRET")})
        res_a, out_a = time_tool.run(home, sec("agent-cc-secret"))
        alog = [json.loads(l) for l in open(a_log)] if os.path.exists(a_log) else []
        tokens = [e for e in alog if e["role"] == "as" and e["path"].startswith("/token")]
        leaked = [e for e in alog if e["has_secret"]]
        res_k, _ = time_tool.run(time_tool.home({"url": rs_url, **cc("agent-cc-secret", client_secret="$KC_SECRET")}), sec("agent-cc-secret"))
        record("CC5 a server naming another authorization server gets 0 token requests and no credential; Keycloak still works pinned", label,
               ok_result(res_k, "current_time") and not tokens and not leaked and not ok_result(res_a, "current_time")
               and SECRETS["agent-cc-secret"] not in out_a, new,
               f"attacker saw {len(alog)} requests, {len(tokens)} token requests, {len(leaked)} carrying a credential; "
               + out_a.strip()[-140:].replace("\n", " "))
        # CC5b (control, so CC5 can fail): pinned to the attacker's own issuer, the same flow does send it the grant.
        home = time_tool.home({"url": f"http://127.0.0.1:{a_rs}/mcp",
                               **cc("agent-cc-secret", client_secret="$KC_SECRET", issuer=f"http://127.0.0.1:{a_as}")})
        time_tool.run(home, sec("agent-cc-secret"))
        alog = [json.loads(l) for l in open(a_log)]
        record("CC5b control: pinned to the attacker's issuer, the credential does reach it", label,
               any(e["role"] == "as" and e["path"].startswith("/token") and e["has_secret"] for e in alog), new)

        # SU1: a CC server whose tool needs an extra scope: 403, step up, success.
        home = write_tool.home({"url": rs_url, **cc("agent-cc-secret", client_secret="$KC_SECRET")})
        m = rs_mark()
        res, out = write_tool.run(home, sec("agent-cc-secret"))
        w = calls(rs_entries(m), "write_note")
        record("SU1 first call 403 insufficient_scope, agent steps up, the call succeeds", label,
               ok_result(res, "written") and [e["status"] for e in w] == [403, 200], new,
               f"write_note statuses={[e['status'] for e in w]} " + (res or {}).get("text", out)[:100].replace("\n", " "))

        # SU3: the AS refuses the extra scope (client may only have mcp:read): one try, clear error, no loop.
        home = write_tool.home({"url": rs_url, **cc("agent-cc-narrow", client_secret="$KC_SECRET")})
        m = rs_mark()
        res, out = write_tool.run(home, sec("agent-cc-narrow"))
        w = calls(rs_entries(m), "write_note")
        text = (res or {}).get("text", "") + " " + out
        record("SU3 AS refuses the scope: the call fails once with a clear error, no loop", label,
               bool(res) and res["is_error"] and [e["status"] for e in w] == [403] and "mcp:write" in text, new,
               f"write_note statuses={[e['status'] for e in w]} " + text.strip()[:160].replace("\n", " "))

        # SU2: an interactive (mcp-login) server: the error names the scope and the command; running it fixes the call.
        home = write_tool.home({"url": rs_url})
        ok, err = write_tool.login(home, issuer)
        res_r, out_r = time_tool.run(home)
        record("SU2a mcp-login against Keycloak (DCR + PKCE, headless form) then a call works", label,
               ok and ok_result(res_r, "current_time"), new, err.strip().splitlines()[-1][:120] if err.strip() else "")
        res, out = write_tool.run(home)
        text = (res or {}).get("text", "") + " " + out
        record("SU2b missing scope: the error names `mcp:write` and `mcp-login kc --scope`", label,
               "mcp:write" in text and "mcp-login kc --scope" in text, new, text.strip()[:200].replace("\n", " "))
        if new:
            ok, err = write_tool.login(home, issuer, ["--scope", "mcp:write"])
            res, out = write_tool.run(home)
            res_t, _ = time_tool.run(home)
            record("SU2c after `mcp-login kc --scope mcp:write` the call succeeds (and the old scope is kept)", label,
                   ok and ok_result(res, "written") and ok_result(res_t, "current_time"), True,
                   (res or {}).get("text", out)[:100].replace("\n", " ") + " | " + err.strip()[-100:].replace("\n", " "))

    bad = [r for r in base.rows if r[2] != r[3]]
    print("\n| assertion | " + " | ".join(bins) + " |\n|---|" + "---|" * len(bins))
    for a in dict.fromkeys(r[0] for r in base.rows):
        cells = []
        for label in bins:
            r = next((x for x in base.rows if x[0] == a and x[1] == label), None)
            cells.append("n/a" if r is None else f"{r[2]}" + ("" if r[2] == r[3] else " (UNEXPECTED)"))
        print(f"| {a} | " + " | ".join(cells) + " |")
    print(f"\nlogs: {LOGDIR}\n{len(base.rows) - len(bad)}/{len(base.rows)} observations as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        for name in containers:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        import signal
        for p in base.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except OSError:
                    pass
        time.sleep(0.5)
        for p in base.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
    sys.exit(code)
