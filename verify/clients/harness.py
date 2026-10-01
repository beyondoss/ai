"""Live harness cells: a real coding agent fixes a failing test in a fixture repo, through the gateway.

Usage: harness.py <harness>[:<mode>]   harness in claude-code | codex | opencode | pi;
       pi's mode is the API it speaks: chat | messages | responses.
Env: VERIFY_BASE (gateway origin), VERIFY_KEY (bai_ key), VERIFY_MODEL.

Each harness runs from its pinned copy in verify/clients/node/node_modules/.bin, with HOME and its
own config directory pointed at a fresh temp dir, so nothing on this machine leaks in. For pi and
opencode, which take models only from a config file, the config is generated from the gateway's own
GET /v1/models (ids, limits, prices) — never written by hand (claim E7).

Prints `VERIFY {json}`: ok (the fixture test passes, the test file is unchanged, and for pi/opencode
the harness's model list equals /v1/models), calls = null (a harness's HTTP calls aren't visible to
us; the Rust cell checks the ledger in aggregate), detail.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
BIN = ROOT / "node" / "node_modules" / ".bin"
BASE = os.environ["VERIFY_BASE"]
KEY = os.environ["VERIFY_KEY"]
MODEL = os.environ["VERIFY_MODEL"]
TIMEOUT = 420

CALC = '''def add(a, b):
    """Return the sum of a and b."""
    return a - b


def scale(xs, k):
    """Multiply every element of xs by k."""
    return [x * k for x in xs]
'''
TEST = '''from calc import add, scale

assert add(2, 3) == 5, f"add(2, 3) == {add(2, 3)}"
assert add(-1, 1) == 0
assert scale([1, 2], 3) == [3, 6]
print("ALL TESTS PASS")
'''
TASK = ("The test in test_calc.py fails. Fix the bug in calc.py without modifying test_calc.py, "
        "then run `python3 test_calc.py` to confirm it prints ALL TESTS PASS.")


def catalog():
    req = urllib.request.Request(f"{BASE}/v1/models", headers={"authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]


def money(v):
    return float(v)


def pi_models(models):
    out = []
    for m in models:
        if "/v1/embeddings" in m["endpoints"]:
            continue
        p = m["pricing"]
        out.append({
            "id": m["id"], "name": m["display_name"],
            "reasoning": "reasoning" in m["capabilities"],
            "input": [x for x in m["input_modalities"] if x in ("text", "image")],
            "contextWindow": m["context_window"], "maxTokens": m["max_output_tokens"],
            "cost": {"input": money(p["input"]), "output": money(p["output"]),
                     "cacheRead": money(p["cache_read"]), "cacheWrite": money(p["cache_write"])},
        })
    return out


def opencode_models(models):
    out = {}
    for m in models:
        if "/v1/embeddings" in m["endpoints"]:
            continue
        p = m["pricing"]
        out[m["id"]] = {
            "name": m["display_name"],
            "limit": {"context": m["context_window"], "output": m["max_output_tokens"]},
            "cost": {"input": money(p["input"]), "output": money(p["output"]),
                     "cache_read": money(p["cache_read"]), "cache_write": money(p["cache_write"])},
            "tool_call": "tools" in m["capabilities"],
            "reasoning": "reasoning" in m["capabilities"],
        }
    return out


def run(cmd, cwd, env):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    return p.returncode, p.stdout[-3000:], p.stderr[-3000:]


def main(spec):
    harness, _, mode = spec.partition(":")
    home = pathlib.Path(tempfile.mkdtemp(prefix=f"verify-{harness}-"))
    work = home / "repo"
    work.mkdir()
    (work / "calc.py").write_text(CALC)
    (work / "test_calc.py").write_text(TEST)
    test_hash = hashlib.sha256(TEST.encode()).hexdigest()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC", "OPENAI", "CLAUDE", "CODEX"))}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local/share"),
               XDG_CACHE_HOME=str(home / ".cache"), XDG_STATE_HOME=str(home / ".local/state"))
    detail = {"harness": harness, "mode": mode or None}
    listed_ok = True

    if harness == "claude-code":
        env.update(ANTHROPIC_BASE_URL=BASE, ANTHROPIC_API_KEY=KEY, DISABLE_TELEMETRY="1",
                   DISABLE_AUTOUPDATER="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL, ANTHROPIC_SMALL_FAST_MODEL=MODEL)
        cmd = [str(BIN / "claude"), "-p", TASK, "--model", MODEL, "--dangerously-skip-permissions",
               "--output-format", "json"]
    elif harness == "codex":
        codex_home = home / ".codex"
        codex_home.mkdir()
        (codex_home / "config.toml").write_text(
            f'model = "{MODEL}"\nmodel_provider = "beyond"\n\n[model_providers.beyond]\nname = "Beyond"\n'
            f'base_url = "{BASE}/v1"\nenv_key = "BEYOND_API_KEY"\nwire_api = "responses"\n')
        env.update(CODEX_HOME=str(codex_home), BEYOND_API_KEY=KEY)
        cmd = [str(BIN / "codex"), "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
               "-m", MODEL, TASK]
    elif harness == "opencode":
        models = catalog()
        cfg = {"$schema": "https://opencode.ai/config.json",
               "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
               "provider": {"beyond": {"npm": "@ai-sdk/openai-compatible", "name": "Beyond",
                                       "options": {"baseURL": f"{BASE}/v1", "apiKey": KEY},
                                       "models": opencode_models(models)}}}
        (work / "opencode.json").write_text(json.dumps(cfg))
        code, out, _ = run([str(BIN / "opencode"), "models", "beyond"], work, env)
        listed = {l.strip().removeprefix("beyond/") for l in out.splitlines() if l.strip().startswith("beyond/")}
        want = set(opencode_models(models))
        listed_ok = listed == want
        detail["models_listed"] = {"harness": len(listed), "catalog": len(want), "missing": sorted(want - listed)[:5]}
        cmd = [str(BIN / "opencode"), "run", "-m", f"beyond/{MODEL}", TASK]
    elif harness == "pi":
        api = {"chat": "openai-completions", "messages": "anthropic-messages", "responses": "openai-responses"}[mode]
        base = BASE if mode == "messages" else f"{BASE}/v1"
        agent_dir = home / ".pi-agent"
        agent_dir.mkdir()
        models = catalog()
        (agent_dir / "models.json").write_text(json.dumps(
            {"providers": {"beyond": {"baseUrl": base, "api": api, "apiKey": KEY, "models": pi_models(models)}}}))
        env.update(PI_CODING_AGENT_DIR=str(agent_dir))
        code, out, err = run([str(BIN / "pi"), "--list-models", "beyond"], work, env)
        listed = {tok for line in out.splitlines() for tok in line.split() if tok in {m["id"] for m in models}}
        want = {m["id"] for m in pi_models(models)}
        listed_ok = listed == want
        detail["models_listed"] = {"harness": len(listed), "catalog": len(want), "missing": sorted(want - listed)[:5]}
        cmd = [str(BIN / "pi"), "-p", "--provider", "beyond", "--model", MODEL, "--no-session", TASK]
    else:
        return False, {"why": f"unknown harness {harness}"}

    code, out, err = run(cmd, work, env)
    test = subprocess.run([sys.executable, "test_calc.py"], cwd=work, capture_output=True, text=True)
    unchanged = hashlib.sha256((work / "test_calc.py").read_bytes()).hexdigest() == test_hash
    detail.update(exit=code, test_rc=test.returncode, test_unchanged=unchanged,
                  stdout_tail=out[-800:], stderr_tail=err[-800:])
    ok = test.returncode == 0 and "ALL TESTS PASS" in test.stdout and unchanged and listed_ok
    shutil.rmtree(home, ignore_errors=True)
    return ok, detail


if __name__ == "__main__":
    try:
        ok, detail = main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}"}
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": None, "detail": detail}, default=str))
