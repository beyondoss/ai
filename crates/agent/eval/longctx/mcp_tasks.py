"""MCP-heavy tasks: short jobs that need tools from three typical MCP servers at once.

Every task runs with the same three servers configured (Playwright, GitHub, filesystem: ~70 tools
between them), so the tool catalog is large whether or not a given task needs all three. That is the
case tool search and Code Mode exist for. Each task has an exact, automatically checked answer:
browser tasks render their data in JavaScript (so the answer is on the page, not in its source text
in a greppable form), GitHub tasks ask about immutable history whose truth is fetched at build time.
"""

from __future__ import annotations

import base64
import json
import os
import random
import subprocess
import urllib.request
from pathlib import Path

SERVERS = {  # pinned
    "playwright": ["-y", "@playwright/mcp@0.0.83", "--headless", "--isolated", "--browser", "chromium"],
    "github": ["-y", "@modelcontextprotocol/server-github@2025.4.8"],
    "filesystem": ["-y", "@modelcontextprotocol/server-filesystem@1.0.0"],
}


def gh_token() -> str:
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()


def gh_api(path: str) -> dict:
    req = urllib.request.Request(f"https://api.github.com/{path}",
                                 headers={"Authorization": f"Bearer {gh_token()}", "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def cache_env(cache: Path) -> dict[str, str]:
    return {"HOME": str(cache), "npm_config_cache": str(cache / "npm"),
            "PLAYWRIGHT_BROWSERS_PATH": str(cache / "ms-playwright")}


def prewarm(cache: Path) -> None:
    """Fetch the pinned servers and Playwright's Chromium into `cache` once, so no run pays for the
    download inside its own wall clock (idempotent)."""
    marker = cache / ".warm"
    want = json.dumps(SERVERS)
    if marker.exists() and marker.read_text() == want:
        return
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **cache_env(cache)}
    for args in SERVERS.values():
        subprocess.run(["npx", *args[:2], "--help"], env=env, cwd=cache, capture_output=True, stdin=subprocess.DEVNULL, timeout=600)
    # The Chromium build that @playwright/mcp's own bundled playwright expects.
    clis = sorted((cache / "npm" / "_npx").glob("*/node_modules/playwright/cli.js"))
    if not clis:
        raise RuntimeError("prewarm: @playwright/mcp did not install playwright")
    for cli in clis:
        subprocess.run(["node", str(cli), "install", "chromium"], env=env, cwd=cache, check=True, timeout=900)
    marker.write_text(want)


def settings(workdir: Path, cache: Path) -> dict:
    """`$HOME/.beyond/settings.json` for a run: the three servers, sharing one npm/browser cache."""
    env = cache_env(cache)
    servers = []
    for name, args in SERVERS.items():
        a = list(args)
        e = dict(env)
        if name == "filesystem":
            a.append(str(workdir))
        if name == "github":
            e["GITHUB_PERSONAL_ACCESS_TOKEN"] = "${GH_TOKEN}"
        servers.append({"name": name, "transport": "stdio", "command": "npx", "args": a, "env": e})
    return {"mcp_servers": servers}


# ---------------------------------------------------------------- tasks

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>
<body><h1>{title}</h1><div id="app">Loading…</div>
<script>
const k = {key};
const raw = atob("{blob}");
let s = ""; for (let i = 0; i < raw.length; i++) s += String.fromCharCode(raw.charCodeAt(i) ^ k);
const data = JSON.parse(s);
{script}
</script></body></html>"""


def _obfuscate(obj, key: int) -> str:
    s = json.dumps(obj)
    return base64.b64encode(bytes(ord(c) ^ key for c in s)).decode()


def browser_table(d: Path) -> tuple[str, str]:
    rng = random.Random(11)
    regions = ["North", "South", "East", "West"]
    rows = [{"id": i, "region": rng.choice(regions), "amount": rng.randint(100, 9999)} for i in range(60)]
    script = """setTimeout(() => {
  let h = "<table id='t'><tr><th>ID</th><th>Region</th><th>Amount</th></tr>";
  for (const r of data) h += `<tr><td>${r.id}</td><td>${r.region}</td><td>${r.amount}</td></tr>`;
  document.getElementById("app").innerHTML = h + "</table>";
}, 400);"""
    (d / "site").mkdir(parents=True, exist_ok=True)
    (d / "site" / "ledger.html").write_text(_PAGE.format(title="Regional ledger", key=73, blob=_obfuscate(rows, 73), script=script))
    want = str(sum(r["amount"] for r in rows if r["region"] == "North"))
    prompt = ("Open {base}/ledger.html in the browser. It renders a table with JavaScript. Add up the Amount "
              "column for every row whose Region is North, and write just that number to `answer.txt` in "
              "this directory.")
    return prompt, want


def browser_click(d: Path) -> tuple[str, str]:
    rng = random.Random(23)
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))
    script = """document.getElementById("app").innerHTML = "<button id='reveal'>Reveal code</button><p id='out'></p>";
document.getElementById("reveal").onclick = () => setTimeout(() => {
  document.getElementById("out").textContent = "Code: " + data.parts.join("");
}, 300);"""
    parts = [code[i:i + 2] for i in range(0, 10, 2)]
    (d / "site").mkdir(parents=True, exist_ok=True)
    (d / "site" / "vault.html").write_text(_PAGE.format(title="Vault", key=91, blob=_obfuscate({"parts": parts}, 91), script=script))
    prompt = ("Open {base}/vault.html in the browser and click the \"Reveal code\" button. Write the code it "
              "shows (just the code, without the \"Code:\" label) to `answer.txt` in this directory.")
    return prompt, code


_PRS = (1551, 1558, 1572)


def github_prs(d: Path) -> tuple[str, str]:
    want = {}
    for n in _PRS:
        p = gh_api(f"repos/pallets/click/pulls/{n}")
        want[str(n)] = {"title": p["title"], "author": p["user"]["login"], "merged": bool(p["merged_at"])}
    prompt = ("Using the GitHub tools, look up pull requests " + ", ".join(f"#{n}" for n in _PRS) + " in pallets/click. "
              "Write `answer.json` in this directory: an object keyed by PR number (as a string), each value "
              "{\"title\": <exact title>, \"author\": <author login>, \"merged\": <true|false>}.")
    return prompt, json.dumps(want, sort_keys=True)


def github_file(d: Path) -> tuple[str, str]:
    c = gh_api("repos/pallets/markupsafe/contents/src/markupsafe/__init__.py?ref=3.0.2")
    src = base64.b64decode(c["content"]).decode()
    names = sorted(line.split("def ", 1)[1].split("(", 1)[0] for line in src.splitlines() if line.startswith("def "))
    prompt = ("Using the GitHub tools, read `src/markupsafe/__init__.py` from pallets/markupsafe at tag 3.0.2. "
              "Then, using the filesystem tools, create `answer.txt` in this directory listing the names of the "
              "module-level functions it defines (lines starting with `def `), one per line, sorted alphabetically.")
    return prompt, "\n".join(names)


MCP_TASKS = {
    "mcp-browser-table": browser_table,
    "mcp-browser-click": browser_click,
    "mcp-github-prs": github_prs,
    "mcp-github-file": github_file,
}


def check(task: str, workdir: Path, want: str) -> bool:
    out = workdir / ("answer.json" if task == "mcp-github-prs" else "answer.txt")
    if not out.exists():
        return False
    got = out.read_text().strip()
    if task == "mcp-github-prs":
        try:
            return json.dumps(json.loads(got), sort_keys=True) == want
        except json.JSONDecodeError:
            return False
    if task == "mcp-github-file":
        return [l.strip() for l in got.splitlines() if l.strip()] == want.splitlines()
    return got == want
