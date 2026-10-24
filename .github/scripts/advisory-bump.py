#!/usr/bin/env python3
"""Fix what a lockfile bump can fix of `cargo audit`'s findings, and say what it cannot.

Usage: python3 -I .github/scripts/advisory-bump.py BODY_FILE [--dry-run]

Runs `cargo audit --json`. For each vulnerable package that has a patched version, runs
`cargo update -p NAME@VERSION` (Cargo takes the newest semver-compatible release, so a patch that
needs a new major or a loosened requirement stays unfixed), then audits again. Writes a markdown
description of the outcome to BODY_FILE: what was bumped, and for each advisory still open its id,
title and the dependency chain (`cargo tree -i`). With --dry-run Cargo.lock is restored afterwards.

Exit status, for the advisories workflow: 0 audit clean, nothing to do; 2 Cargo.lock changed and
the audit is now clean; 3 Cargo.lock changed, some advisories remain; 4 Cargo.lock unchanged,
advisories remain (a human has to act); anything else is an unexpected failure.
"""

import json
import shutil
import subprocess
import sys


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def audit():
    p = run("cargo", "audit", "--json")
    try:
        return json.loads(p.stdout)["vulnerabilities"]["list"]
    except (ValueError, KeyError):
        sys.exit(f"cargo audit produced no usable JSON (exit {p.returncode}):\n{p.stderr}")


def key(v):
    return v["advisory"]["id"], v["package"]["name"], v["package"]["version"]


def tree(name, version):
    p = run("cargo", "tree", "-i", f"{name}@{version}", "-e", "normal,build")
    return (p.stdout or p.stderr).strip()


def main():
    body_path = sys.argv[1]
    dry = "--dry-run" in sys.argv[2:]
    original = open("Cargo.lock").read()
    try:
        before = audit()
        if not before:
            open(body_path, "w").write("`cargo audit` is clean.\n")
            return 0
        targets = sorted({(v["package"]["name"], v["package"]["version"])
                          for v in before if v["versions"]["patched"]})
        for name, version in targets:
            print(f"+ cargo update -p {name}@{version}", flush=True)
            print(run("cargo", "update", "-p", f"{name}@{version}").stderr.strip(), flush=True)
        after = audit()
        changed = open("Cargo.lock").read() != original
        old = {(k[1], k[2]) for k in map(key, before)}
        # An advisory on a package bumped to a release that is still vulnerable is progress, not a fix.
        partial = {(k[0], k[1]) for k in map(key, after) if (k[1], k[2]) not in old}
        fixed = {k for k in map(key, before) if k not in set(map(key, after)) and k[:2] not in partial}
        out = []
        if changed:
            out.append("Bumps vulnerable packages in `Cargo.lock` to patched releases "
                       "(`cargo update -p NAME@VERSION`, semver-compatible only).\n")
            out += [f"- fixed {i}: {n} {v}" for i, n, v in sorted(fixed)]
        if after:
            out.append("\nNot fixable by a lockfile bump (needs a dependency requirement change, "
                       "a major bump, or an ignore with an exposure assessment in "
                       ".cargo/audit.toml):\n")
            for v in after:
                out.append(f"- {v['advisory']['id']}: {v['package']['name']} {v['package']['version']}, "
                           f"{v['advisory']['title']}; patched: "
                           f"{', '.join(v['versions']['patched']) or 'none'}")
            for name, version in sorted({(k[1], k[2]) for k in map(key, after)}):
                out.append(f"\n`cargo tree -i {name}@{version}`:\n\n```\n{tree(name, version)}\n```")
        text = "\n".join(out) + "\n"
        open(body_path, "w").write(text)
        print(text)
        return (3 if after else 2) if changed else 4
    finally:
        if dry:
            open("Cargo.lock", "w").write(original)


if __name__ == "__main__":
    sys.exit(main())
