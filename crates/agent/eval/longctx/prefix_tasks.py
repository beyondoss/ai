"""Prompt-prefix tasks: short jobs whose answer lives in what the agent injects before turn one.

The context-plan changes budget that prefix: the skills listing (names and short descriptions, no
paths), the memory index (4 KB in the prompt), project instruction files (32 KiB each). Each task
here puts the one fact it needs where a budget can cut it, inside a realistically crowded HOME
(two dozen skills, a full memory index, a long AGENTS.md), so a regression in "the model still finds
what was budgeted away" shows up as a failed task rather than as a smaller token count.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

_DECOY_SKILLS = [
    ("pdf-forms", "Fill, flatten and validate PDF forms. Use when a task mentions AcroForm fields, form filling, or producing a signed PDF from a template."),
    ("terraform-plan-review", "Review a Terraform plan for destructive changes, drift and missing tags before apply. Use when the user pastes plan output or asks whether an apply is safe."),
    ("sql-migration", "Write reversible SQL migrations with an explicit down step, lock-timeout guards and a backfill plan. Use for schema changes on large Postgres tables."),
    ("incident-timeline", "Turn chat logs and alerts into a minute-by-minute incident timeline with detection, mitigation and resolution markers. Use after an outage."),
    ("k8s-debug", "Diagnose CrashLoopBackOff, OOMKilled and pending pods from kubectl describe/logs output. Use when a deployment won't become ready."),
    ("perf-flamegraph", "Capture and read a CPU flamegraph with perf or py-spy and name the hottest frames. Use when something is slow and nobody knows why."),
    ("dependency-bump", "Upgrade one dependency across a monorepo, run the affected tests and summarize breaking changes from its changelog. Use for version bumps."),
    ("api-docs", "Generate reference docs for an HTTP API from its OpenAPI spec, with one runnable example per endpoint. Use when docs are missing or stale."),
    ("code-review", "Review a diff for correctness bugs first, then for clarity. Report findings ranked by severity with file:line references. Use on any PR review request."),
    ("regex-builder", "Build and test a regular expression against positive and negative examples, explaining each part. Use when a user needs a pattern."),
    ("csv-cleanup", "Normalize messy CSV files: encodings, headers, delimiters, date formats and duplicate rows. Use before loading data anywhere."),
    ("security-headers", "Audit a web app's HTTP response headers (CSP, HSTS, frame options) and propose a policy. Use for security hardening of web endpoints."),
    ("git-bisect", "Drive git bisect with a scripted test to find the commit that introduced a regression. Use when something used to work and now doesn't."),
    ("docker-slim", "Shrink a Docker image with multi-stage builds, distroless bases and layer ordering. Use when an image is too large or slow to pull."),
    ("load-test", "Write and run a k6 load test with ramp stages and thresholds, then summarize p50/p95/p99. Use before a launch or after a perf change."),
    ("oncall-handoff", "Summarize the week's pages, open incidents and risky deploys into an on-call handoff note. Use at the end of an on-call shift."),
    ("type-coverage", "Raise Python type coverage with mypy --strict on one package at a time, fixing the root causes of Any. Use when typing a codebase."),
    ("feature-flag", "Add a feature flag with a default-off rollout, a kill switch and a cleanup ticket. Use when shipping risky behavior behind a flag."),
    ("i18n-extract", "Extract user-facing strings into message catalogs and flag concatenated or untranslatable strings. Use when preparing a UI for translation."),
    ("cron-audit", "List every scheduled job, its owner and its failure alerting, and flag jobs nobody would notice failing. Use for ops hygiene reviews."),
    ("schema-diff", "Compare two JSON Schemas or protobuf files and classify every change as breaking or compatible. Use before publishing an API change."),
    ("log-query", "Write a log query (LogQL, Splunk or CloudWatch Insights) from a plain-language question and explain the result. Use when searching logs."),
]

_CHANGELOG = ("changelog-entry", "Write a CHANGELOG.md entry in this team's house format. Use whenever a task asks to add or update a changelog, release notes or a version entry.",
              """# Changelog entries

House format, always:

1. Heading `## [<version>] - <YYYY-MM-DD>` (square brackets around the version).
2. Group bullets under `### Added`, `### Fixed`, `### Changed` (omit empty groups).
3. The file's last line must be exactly:
   `<!-- changelog-format: house-v3 -->`
   (keep a single copy of it at the very end of the file).
""")

_NAMING = ("api-naming", "Naming rules for new HTTP endpoints in the billing API. Use when designing or naming an endpoint, route or path.",
           """# Billing API naming

- Collections are plural kebab-case nouns under `/v3/`.
- A list endpoint is `GET /v3/<collection>:list` (the `:list` custom verb is mandatory; a bare
  `GET /v3/<collection>` is reserved for the streaming export).
- Write the method and path on one line, separated by one space.
""")


def _skills(home: Path) -> None:
    root = home / ".claude" / "skills"
    for name, desc in _DECOY_SKILLS:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\nFollow the steps for {name}.\n")
    for name, desc, body in (_CHANGELOG, _NAMING):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\n{body}")


def _memory(home: Path, work: Path) -> None:
    """A ~18 KB MEMORY.md (under the old 25 KB read cap, far over the new 4 KB prompt cap) with the
    one fact the task needs near its end."""
    enc = str(work.resolve()).replace("/", "-")
    mem = home / ".claude" / "projects" / enc / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    rng = random.Random(5)
    topics = ["deploy", "ci", "auth", "billing", "search", "cache", "queue", "metrics", "frontend", "mobile"]
    lines = ["# Project memory", ""]
    for i in range(170):
        t = rng.choice(topics)
        lines.append(f"- [{t}-note-{i}]({t}-note-{i}.md) — {t}: decision {i} about the {t} path; "
                     f"see the linked note for the reasoning and the ticket that followed it")
        if i == 150:
            lines.append("- [staging-db](staging-db.md) — the staging Postgres listens on port 6543, not 5432 "
                         "(pgbouncer sits in front); connect with sslmode=require")
    (mem / "MEMORY.md").write_text("\n".join(lines) + "\n")
    (mem / "staging-db.md").write_text("Staging Postgres: port 6543 behind pgbouncer, sslmode=require.\n")


def _agents_md(work: Path) -> None:
    """A ~60 KiB AGENTS.md whose one task-relevant rule sits past 32 KiB."""
    rng = random.Random(9)
    areas = ["Error handling", "Logging", "Testing", "Naming", "Imports", "Configuration", "Concurrency",
             "HTTP clients", "Databases", "Serialization", "Docs", "Release"]
    out = ["# Engineering guidelines", ""]
    for i in range(120):
        a = rng.choice(areas)
        out.append(f"## {a} rule {i}")
        out.append(f"When working on {a.lower()}, prefer the established helper over a new one, keep the change "
                   f"small, and explain the trade-off in the PR description. Rule {i} exists because a past change "
                   f"in this area caused an incident that took a day to unwind; reviewers will ask about it.")
        out.append("")
        if i == 90:
            out.append("## Module headers")
            out.append("Every new Python module must begin with this exact first line: `# owner: platform-team`")
            out.append("")
    work.mkdir(parents=True, exist_ok=True)
    (work / "AGENTS.md").write_text("\n".join(out))


def ctx_skill_implicit(home: Path, work: Path) -> tuple[str, object]:
    _skills(home)
    (work / "CHANGELOG.md").write_text("# Changelog\n")
    prompt = ("Add a changelog entry for version 2.4.0, dated 2026-10-11, covering two changes: we fixed the retry "
              "loop that gave up after one attempt, and we added a --dry-run flag.")

    def check() -> bool:
        p = work / "CHANGELOG.md"
        if not p.exists():
            return False
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        return any(l.strip() == "## [2.4.0] - 2026-10-11" for l in lines) and lines[-1].strip() == "<!-- changelog-format: house-v3 -->"
    return prompt, check


def ctx_skill_named(home: Path, work: Path) -> tuple[str, object]:
    _skills(home)
    prompt = ("Use the api-naming skill to name a new endpoint that lists invoices. Write the method and path to "
              "`answer.txt`.")
    return prompt, lambda: (work / "answer.txt").exists() and (work / "answer.txt").read_text().strip() == "GET /v3/invoices:list"


def ctx_memory_tail(home: Path, work: Path) -> tuple[str, object]:
    _skills(home)
    _memory(home, work)
    prompt = "Which port does the staging database listen on? Write just the number to `answer.txt`."
    return prompt, lambda: (work / "answer.txt").exists() and (work / "answer.txt").read_text().strip() == "6543"


def ctx_agents_tail(home: Path, work: Path) -> tuple[str, object]:
    _skills(home)
    _agents_md(work)
    prompt = ("Create a Python module `util/slugify.py` with a function `slugify(s: str) -> str` that lowercases "
              "the string and replaces runs of whitespace with single hyphens.")

    def check() -> bool:
        p = work / "util" / "slugify.py"
        if not p.exists() or p.read_text().splitlines()[:1] != ["# owner: platform-team"]:
            return False
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'util'); import slugify; "
                            "assert slugify.slugify('Hello  Big World') == 'hello-big-world'"], cwd=work)
        return r.returncode == 0
    return prompt, check


PREFIX_TASKS = {
    "ctx-skill-implicit": ctx_skill_implicit,
    "ctx-skill-named": ctx_skill_named,
    "ctx-memory-tail": ctx_memory_tail,
    "ctx-agents-tail": ctx_agents_tail,
}
