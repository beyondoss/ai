"""Live session cells (SES-*) for coding agents: one harness session, through the gateway, that
outlives a provider (SES-1), changes model and dialect between steps (SES-2), or is resumed from
its own stateless transcript (SES-3).

Usage: session_harness.py <scenario>, with harness.py's environment plus VERIFY_MODELS for the
switching scenarios (`model=provider,...`; step k uses entry k mod n).

  claude-code-failover   Claude Code fixes the fixture; the Rust trial kills the primary's proxy
                         after a few requests, so the task finishes on the fallback.
  pi-failover            the same with pi speaking Messages.
  pi-switch              pi: step 1 fixes the fixture on model A (Messages), step 2 continues the
                         same session on model B (Chat), step 3 back on A (Messages); steps 2 and 3
                         can only succeed from the session's history.
  opencode-switch        the same three steps with `opencode run --continue -m`.
  codex-resume           Codex (store:false, encrypted reasoning) fixes the fixture, then
                         `codex exec resume --last` continues it from Codex's own transcript.

Each harness runs from verify/clients/node with HOME in a fresh temp dir (see harness.py). Prints
`VERIFY {json}`: ok, calls = null (the Rust trial checks the ledger in aggregate), and detail.steps,
the model each step ran on, in order, so the trial can hold the ledger's rows to the steps.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as h  # noqa: E402

CODE = "ZEPHYR-" + os.urandom(2).hex().upper()
RECALL_B = ("Earlier in this session you fixed a bug in calc.py. Write the name of the function you fixed, and "
            "nothing else, into a new file fixed.txt.")
RECALL_A = ("Append a second line to fixed.txt containing only our project codename from my first "
            "message. Do not change the first line.")


def plan():
    return [x.split("=")[0] for x in os.environ.get("VERIFY_MODELS", f"{h.MODEL}=").split(",")]


def setup(name):
    home = pathlib.Path(tempfile.mkdtemp(prefix=f"verify-ses-{name}-"))
    work = home / "repo"
    work.mkdir()
    (work / "calc.py").write_text(h.CALC)
    (work / "test_calc.py").write_text(h.TEST)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC", "OPENAI", "CLAUDE", "CODEX"))}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local/share"),
               XDG_CACHE_HOME=str(home / ".cache"), XDG_STATE_HOME=str(home / ".local/state"))
    return home, work, env


def fixture_ok(work):
    test = subprocess.run([sys.executable, "test_calc.py"], cwd=work, capture_output=True, text=True)
    unchanged = hashlib.sha256((work / "test_calc.py").read_bytes()).hexdigest() == hashlib.sha256(
        h.TEST.encode()).hexdigest()
    return test.returncode == 0 and "ALL TESTS PASS" in test.stdout and unchanged


def step(cmd, work, env, steps, model):
    code, out, err = h.run(cmd, work, env)
    sessions = sorted({str(e.get("sessionID")) for e in h.json_lines(out) if e.get("sessionID")})
    steps.append({"model": model, "exit": code, "sessions": sessions, "stdout_tail": out[-400:],
                  "stderr_tail": err[-400:]})
    return code


def recall_ok(work):
    try:
        lines = [x.strip() for x in (work / "fixed.txt").read_text().splitlines() if x.strip()]
    except OSError:
        return False, None
    return len(lines) >= 2 and "add" in lines[0] and CODE in lines[1], lines


def main(scenario):
    home, work, env = setup(scenario)
    steps, detail = [], {"scenario": scenario, "code": CODE}
    task = f"{h.TASK} Also remember our project codename for later: {CODE}."
    if scenario == "claude-code-failover":
        env.update(ANTHROPIC_BASE_URL=h.BASE, ANTHROPIC_API_KEY=h.KEY, DISABLE_TELEMETRY="1", DISABLE_AUTOUPDATER="1",
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", ANTHROPIC_DEFAULT_HAIKU_MODEL=h.MODEL,
                   ANTHROPIC_SMALL_FAST_MODEL=h.MODEL)
        step([str(h.BIN / "claude"), "-p", h.TASK, "--model", h.MODEL, "--dangerously-skip-permissions",
              "--output-format", "json"], work, env, steps, h.MODEL)
        ok = fixture_ok(work)
    elif scenario == "pi-failover":
        agent = home / ".pi-agent"
        agent.mkdir()
        (agent / "models.json").write_text(json.dumps({"providers": {"beyond": {
            "baseUrl": h.BASE, "api": "anthropic-messages", "apiKey": h.KEY, "models": h.pi_models(h.catalog())}}}))
        env.update(PI_CODING_AGENT_DIR=str(agent))
        step([str(h.BIN / "pi"), "-p", "--mode", "json", "--provider", "beyond", "--model", h.MODEL, "--no-session",
              h.TASK], work, env, steps, h.MODEL)
        ok = fixture_ok(work)
    elif scenario == "pi-switch":
        a, b = plan()[:2]
        agent = home / ".pi-agent"
        agent.mkdir()
        models = h.pi_models(h.catalog())
        (agent / "models.json").write_text(json.dumps({"providers": {
            "beyond-messages": {"baseUrl": h.BASE, "api": "anthropic-messages", "apiKey": h.KEY, "models": models},
            "beyond-chat": {"baseUrl": f"{h.BASE}/v1", "api": "openai-completions", "apiKey": h.KEY, "models": models}}}))
        env.update(PI_CODING_AGENT_DIR=str(agent))
        sessions = home / "sessions"
        pi = [str(h.BIN / "pi"), "-p", "--mode", "json", "--session-dir", str(sessions)]
        step(pi + ["--provider", "beyond-messages", "--model", a, task], work, env, steps, a)
        step(pi + ["--continue", "--provider", "beyond-chat", "--model", b, RECALL_B], work, env, steps, b)
        step(pi + ["--continue", "--provider", "beyond-messages", "--model", a, RECALL_A], work, env, steps, a)
        detail["session_files"] = [str(x.name) for x in sessions.rglob("*.jsonl")]
        good, detail["fixed_txt"] = recall_ok(work)
        ok = fixture_ok(work) and good and len(detail["session_files"]) == 1
    elif scenario == "opencode-switch":
        a, b = plan()[:2]
        cfg = {"$schema": "https://opencode.ai/config.json",
               "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
               "provider": {"beyond": {"npm": "@ai-sdk/openai-compatible", "name": "Beyond",
                                       "options": {"baseURL": f"{h.BASE}/v1", "apiKey": h.KEY},
                                       "models": h.opencode_models(h.catalog())}}}
        (work / "opencode.json").write_text(json.dumps(cfg))
        oc = [str(h.BIN / "opencode"), "run", "--format", "json", "--title", "fix calc"]
        step(oc + ["-m", f"beyond/{a}", task], work, env, steps, a)
        step(oc + ["--continue", "-m", f"beyond/{b}", RECALL_B], work, env, steps, b)
        step(oc + ["--continue", "-m", f"beyond/{a}", RECALL_A], work, env, steps, a)
        ids = {i for s in steps for i in s["sessions"]}
        good, detail["fixed_txt"] = recall_ok(work)
        ok = fixture_ok(work) and good and len(ids) == 1
    elif scenario == "codex-resume":
        codex_home = home / ".codex"
        codex_home.mkdir()
        (codex_home / "config.toml").write_text(
            f'model = "{h.MODEL}"\nmodel_provider = "beyond"\n\n[model_providers.beyond]\nname = "Beyond"\n'
            f'base_url = "{h.BASE}/v1"\nenv_key = "BEYOND_API_KEY"\nwire_api = "responses"\n')
        env.update(CODEX_HOME=str(codex_home), BEYOND_API_KEY=h.KEY)
        codex = [str(h.BIN / "codex"), "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]
        step(codex + ["-m", h.MODEL, task], work, env, steps, h.MODEL)
        step(codex + ["resume", "--last", "Write only our project codename from my first message into a new file "
                      "code.txt."], work, env, steps, h.MODEL)
        try:
            detail["code_txt"] = (work / "code.txt").read_text().strip()
        except OSError:
            detail["code_txt"] = None
        ok = fixture_ok(work) and CODE in (detail["code_txt"] or "")
    else:
        return False, {"why": f"unknown scenario {scenario}"}
    detail["steps"] = steps
    shutil.rmtree(home, ignore_errors=True)
    return ok, detail


if __name__ == "__main__":
    try:
        ok, detail = main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        ok, detail = False, {"exception": f"{type(e).__name__}: {e}"}
    print("VERIFY " + json.dumps({"ok": bool(ok), "calls": None, "detail": detail}, default=str))
