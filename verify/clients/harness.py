"""Live harness cells: a real coding agent fixes a failing test in a fixture repo, through the gateway.

Usage: harness.py <harness>[:<mode>]   harness in claude-code | codex | opencode | pi;
       pi's mode is the API it speaks: chat | messages | responses.
Env: VERIFY_BASE (gateway origin), VERIFY_KEY (bai_ key), VERIFY_MODEL.

Each harness runs from its pinned copy in verify/clients/node/node_modules/.bin, with HOME and its
own config directory pointed at a fresh temp dir, so nothing on this machine leaks in. For pi and
opencode, which take models only from a config file, the config is generated from the gateway's own
GET /v1/models (ids, limits, prices) — never written by hand (claim E7).

Prints `VERIFY {json}`: ok (the fixture test passes and the test file is unchanged), calls = null
(a harness's HTTP calls aren't visible to us; the Rust cell checks the ledger in aggregate), detail.
For claim E7 the detail also carries what the harness itself shows, read back from the harness:

- `listing`: the models it lists, compared with /v1/models (`ok`, and what differs).
- `cost`: the session cost it displays (`displayed`, USD, or null with `why`) and the card prices
  of VERIFY_MODEL (`pricing`, USD per million tokens). The Rust cell prices the ledger rows at that
  card and compares.

What each harness can show, as of the pinned versions:

| harness     | lists models from the gateway            | displays session cost                    |
| ----------- | ---------------------------------------- | ---------------------------------------- |
| claude-code | /v1/models, ids matching /claude|anthropic/i, no limits or prices (CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY) | total_cost_usd, from its own price table; comparable only where it knows the model (costBasis "list") |
| opencode    | generated config; `models --verbose` shows limits and prices | per-step cost in `--format json`, from the config prices |
| pi          | generated models.json; `--list-models` shows limits | per-message cost in `--mode json`, from models.json prices |
| codex       | no: only its bundled catalog, or a model_catalog_json that needs Codex's own system prompt per model | no |
"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from decimal import ROUND_HALF_UP, Decimal

ROOT = pathlib.Path(__file__).resolve().parent
BIN = ROOT / "node" / "node_modules" / ".bin"
BASE = os.environ["VERIFY_BASE"]
KEY = os.environ["VERIFY_KEY"]
MODEL = os.environ["VERIFY_MODEL"]
TIMEOUT = 420
# Extra arguments for the harness command line (`harness_long.py cell` sets them, e.g. pi's
# `--thinking low`).
EXTRA_ARGS = []

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


def chat_models(models):
    """The rows a coding agent can drive: everything but embeddings."""
    return [m for m in models if "/v1/embeddings" not in m["endpoints"]]


def prices(m):
    """A row's card prices, USD per million tokens."""
    return {k: float(v) for k, v in m["pricing"].items()}


def pi_models(models):
    out = []
    for m in chat_models(models):
        p = prices(m)
        out.append({
            "id": m["id"], "name": m["display_name"],
            "reasoning": "reasoning" in m["capabilities"],
            "input": [x for x in m["input_modalities"] if x in ("text", "image")],
            "contextWindow": m["context_window"], "maxTokens": m["max_output_tokens"],
            "cost": {"input": p["input"], "output": p["output"],
                     "cacheRead": p["cache_read"], "cacheWrite": p["cache_write"]},
        })
    return out


def opencode_models(models):
    out = {}
    for m in chat_models(models):
        p = prices(m)
        out[m["id"]] = {
            "name": m["display_name"],
            "limit": {"context": m["context_window"], "output": m["max_output_tokens"]},
            "cost": {"input": p["input"], "output": p["output"],
                     "cache_read": p["cache_read"], "cache_write": p["cache_write"]},
            "modalities": {"input": [x for x in m["input_modalities"] if x in ("text", "image")],
                           "output": ["text"]},
            "tool_call": "tools" in m["capabilities"],
            "reasoning": "reasoning" in m["capabilities"],
        }
    return out


def pi_count(n):
    """A count as `pi --list-models` prints it (its formatTokenCount): 200K, 16.4K, 1M, 1.0M, 8192."""
    for unit, size in (("M", 1_000_000), ("K", 1_000)):
        if n >= size:
            v = n / size
            if v % 1 == 0:
                return f"{int(v)}{unit}"
            # JS toFixed(1): the float's exact value, rounded half up.
            return f"{Decimal(v).quantize(Decimal('0.1'), ROUND_HALF_UP)}{unit}"
    return str(n)


def listing(want, listed, mismatch=()):
    """Compare a harness's model ids (and any per-model card mismatches) with the catalog's."""
    return {"ok": listed == want and not mismatch, "harness": len(listed), "catalog": len(want),
            "missing": sorted(want - listed)[:5], "extra": sorted(listed - want)[:5],
            "card_mismatch": list(mismatch)[:5]}


def opencode_verbose(text):
    """`opencode models <provider> --verbose`: each `provider/id` line is followed by its JSON."""
    out, dec, i = {}, json.JSONDecoder(), 0
    for m in re.finditer(r"^beyond/(\S+)\s*$", text, re.M):
        if m.start() < i:
            continue
        start = text.index("{", m.end())
        obj, i = dec.raw_decode(text, start)
        out[m.group(1)] = obj
    return out


def json_lines(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def run(cmd, cwd, env):
    # stdin closed: Codex (and others) read a non-TTY stdin as the prompt and would wait forever.
    # PWD set to cwd, as a shell would: `opencode run` roots its session at $PWD, not the process
    # cwd, so an inherited PWD silently drops the repo's opencode.json (ProviderModelNotFoundError).
    # Output to files, not pipes: bun (opencode) exits without draining a full pipe, so under load
    # a large stdout (`opencode models --verbose` is ~100 KB) came back cut off mid-JSON.
    env = {**env, "PWD": str(cwd)}
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=err, timeout=TIMEOUT,
                           stdin=subprocess.DEVNULL)
        out.seek(0)
        err.seek(0)
        return p.returncode, out.read(), err.read()


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
    models = catalog()
    card = {m["id"]: m for m in models}[MODEL]
    detail = {"harness": harness, "mode": mode or None}
    cost = {"displayed": None, "why": None, "pricing": prices(card)}

    if harness == "claude-code":
        # Gateway model discovery is opt-in; with it, Claude Code GETs /v1/models at startup and
        # caches the ids it keeps (its own filter: /(claude|anthropic)/i) for its /model picker.
        env.update(ANTHROPIC_BASE_URL=BASE, ANTHROPIC_API_KEY=KEY, DISABLE_TELEMETRY="1",
                   DISABLE_AUTOUPDATER="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                   CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY="1",
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
        cost["why"] = "Codex displays no session cost"
    elif harness == "opencode":
        cards = opencode_models(models)
        cfg = {"$schema": "https://opencode.ai/config.json",
               "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
               "provider": {"beyond": {"npm": "@ai-sdk/openai-compatible", "name": "Beyond",
                                       "options": {"baseURL": f"{BASE}/v1", "apiKey": KEY},
                                       "models": cards}}}
        (work / "opencode.json").write_text(json.dumps(cfg))
        _, out, _ = run([str(BIN / "opencode"), "models", "beyond", "--verbose"], work, env)
        shown = opencode_verbose(out)
        mismatch = []
        for mid, s in shown.items():
            c = cards.get(mid)
            if c is None:
                continue
            got = (s["limit"]["context"], s["limit"]["output"], s["cost"]["input"], s["cost"]["output"],
                   s["cost"]["cache"]["read"], s["cost"]["cache"]["write"])
            want = (c["limit"]["context"], c["limit"]["output"], c["cost"]["input"], c["cost"]["output"],
                    c["cost"]["cache_read"], c["cost"]["cache_write"])
            if any(abs(g - w) > 1e-9 for g, w in zip(got, want)):
                mismatch.append(f"{mid}: shows {got}, card {want}")
        detail["listing"] = listing(set(cards), set(shown), mismatch)
        # --format json: one event per line; each step_finish carries the cost opencode charges the
        # session for that step. --title: without it opencode spends an extra call on a title that
        # its session cost never includes, so its displayed cost could not equal the bill.
        cmd = [str(BIN / "opencode"), "run", "--print-logs", "--format", "json", "--title", "fix calc",
               "-m", f"beyond/{MODEL}", TASK]
    elif harness == "pi":
        api = {"chat": "openai-completions", "messages": "anthropic-messages", "responses": "openai-responses"}[mode]
        base = BASE if mode == "messages" else f"{BASE}/v1"
        agent_dir = home / ".pi-agent"
        agent_dir.mkdir()
        (agent_dir / "models.json").write_text(json.dumps(
            {"providers": {"beyond": {"baseUrl": base, "api": api, "apiKey": KEY, "models": pi_models(models)}}}))
        env.update(PI_CODING_AGENT_DIR=str(agent_dir))
        _, out, err = run([str(BIN / "pi"), "--list-models", "beyond"], work, env)
        # pi prints a table (to stderr): provider, model, context, max-out, thinking, images.
        listed, mismatch = set(), []
        cards = {m["id"]: m for m in pi_models(models)}
        for line in (out + "\n" + err).splitlines():
            cols = line.split()
            if len(cols) >= 6 and cols[0] == "beyond" and cols[1] in cards:
                listed.add(cols[1])
                c = cards[cols[1]]
                want = [pi_count(c["contextWindow"]), pi_count(c["maxTokens"]),
                        "yes" if c["reasoning"] else "no", "yes" if "image" in c["input"] else "no"]
                if cols[2:6] != want:
                    mismatch.append(f"{cols[1]}: shows {cols[2:6]}, card {want}")
        detail["listing"] = listing(set(cards), listed, mismatch)
        # --mode json: one event per line; each assistant message_end carries its usage and the
        # cost pi computed from the models.json prices.
        cmd = [str(BIN / "pi"), "-p", "--mode", "json", "--provider", "beyond", "--model", MODEL,
               "--no-session", TASK]
    else:
        return False, {"why": f"unknown harness {harness}"}

    code, out, err = run(cmd + EXTRA_ARGS, work, env)

    if harness == "claude-code":
        result = next((e for e in json_lines(out) if e.get("type") == "result"), {})
        usage = result.get("modelUsage") or {}
        basis = {m: u.get("costBasis") for m, u in usage.items()}
        cost["by_model"] = {m: {k: u.get(k) for k in ("inputTokens", "outputTokens", "cacheReadInputTokens",
                                                       "cacheCreationInputTokens", "costUSD", "costBasis")}
                            for m, u in usage.items()}
        if "total_cost_usd" not in result:
            cost["why"] = "no result event with total_cost_usd"
        elif set(basis) != {MODEL} or any(b != "list" for b in basis.values()):
            # Claude Code prices from its own table; a model it doesn't know gets a placeholder
            # price (costBasis "unknown"), which no bill could match.
            cost["why"] = f"Claude Code's own price table doesn't know the model: costBasis {basis}"
        else:
            cost["displayed"] = result["total_cost_usd"]
        cached = home / ".claude" / "cache" / "gateway-models.json"
        try:
            found = {m["id"] for m in json.loads(cached.read_text())["models"]}
        except (OSError, ValueError, KeyError):
            found = set()
        want = {m["id"] for m in models if re.search("claude|anthropic", m["id"], re.I)}
        detail["listing"] = listing(want, found)
    elif harness == "opencode":
        steps = [e["part"] for e in json_lines(out) if e.get("type") == "step_finish"]
        cost["steps"] = len(steps)
        if steps:
            cost["displayed"] = sum(s.get("cost", 0) for s in steps)
        else:
            cost["why"] = "no step_finish events"
    elif harness == "pi":
        msgs = [e["message"] for e in json_lines(out)
                if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant"]
        cost["messages"] = len(msgs)
        if msgs:
            cost["displayed"] = sum(m["usage"]["cost"]["total"] for m in msgs)
        else:
            cost["why"] = "no assistant message_end events"
    detail["cost"] = cost

    test = subprocess.run([sys.executable, "test_calc.py"], cwd=work, capture_output=True, text=True)
    unchanged = hashlib.sha256((work / "test_calc.py").read_bytes()).hexdigest() == test_hash
    detail.update(exit=code, test_rc=test.returncode, test_unchanged=unchanged,
                  stdout_tail=out[-800:], stderr_tail=err[-800:])
    ok = test.returncode == 0 and "ALL TESTS PASS" in test.stdout and unchanged
    shutil.rmtree(home, ignore_errors=True)
    return ok, detail


if __name__ == "__main__":
    try:
        ok, detail = main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}"}
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": None, "detail": detail}, default=str))
