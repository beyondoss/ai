#!/usr/bin/env python3
"""Fix what a lockfile bump can fix of `cargo audit`'s findings, and say what it cannot.

Usage: python3 -I .github/scripts/advisory-bump.py BODY_FILE [--dry-run]

Runs `cargo audit --json`. For each vulnerable package that has a patched version, runs
`cargo update -p NAME@VERSION --precise MIN` where MIN is the lowest patched release semver-compatible
with VERSION, so a compromised newer release cannot ride in with the fix (a plain `cargo update -p`
takes the newest one). `--precise` takes exactly one package, so the bumps run one at a time, in
rounds: one that Cargo rejects (rustls needs a newer rustls-webpki, say) is retried after the others
have moved. What still fails falls back to a plain `cargo update -p NAME@VERSION`, and the PR body
says so. A patch that needs a new major or a loosened requirement stays unfixed. Then audits again. Writes a markdown
description of the outcome to BODY_FILE: what was bumped, and for each advisory still open its id,
title and the dependency chain (`cargo tree -i`). With --dry-run Cargo.lock is restored afterwards.

Exit status, for the advisories workflow: 0 audit clean, nothing to do; 2 Cargo.lock changed and
the audit is now clean; 3 Cargo.lock changed, some advisories remain; 4 Cargo.lock unchanged,
advisories remain (a human has to act); anything else is an unexpected failure.
"""

import json
import re
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


def parse(v):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v)
    return tuple(map(int, m.groups())) if m else None


def compatible(cur, new):
    """Cargo's caret rule: the same leftmost non-zero component."""
    if new <= cur:
        return False
    if cur[0]:
        return new[0] == cur[0]
    if cur[1]:
        return new[:2] == cur[:2]
    return new == cur


def min_patched(version, patched):
    """The lowest lower bound among the advisory's patched ranges that `version` can move to."""
    cur = parse(version)
    if cur is None:
        return None
    found = []
    for rng in patched:
        for comp in rng.split(","):
            m = re.fullmatch(r"\s*(>=|\^|~|=)?\s*(\d+\.\d+\.\d+)\s*", comp)
            # `<` / `>` / a pre-release bound is no usable lower bound; only the first comparator of
            # a range is its lower bound.
            if m and parse(m.group(2)) and compatible(cur, parse(m.group(2))):
                found.append(parse(m.group(2)))
            break
    return ".".join(map(str, min(found))) if found else None


def key(v):
    return v["advisory"]["id"], v["package"]["name"], v["package"]["version"]


def tree(name, version):
    p = run("cargo", "tree", "-i", f"{name}@{version}", "--workspace", "--target", "all")
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
        # One target per package version: it must reach the highest of its advisories' minimums.
        wanted = {}
        for v in before:
            if v["versions"]["patched"]:
                wanted.setdefault((v["package"]["name"], v["package"]["version"]), []).append(
                    min_patched(v["package"]["version"], v["versions"]["patched"]))
        pending = []
        fallback_notes = []
        for (n, v), mins in sorted(wanted.items()):
            if None in mins:
                fallback_notes.append((n, v, "no patched range with a lower bound compatible with the locked version"))
            else:
                pending.append((n, v, max(mins, key=parse)))
        # Rounds: a precise bump can need another package to have moved first, so retry the
        # rejected ones until a whole round makes no progress.
        failed = pending
        while failed:
            rejected = []
            for n, v, m in failed:
                print(f"+ cargo update -p {n}@{v} --precise {m}", flush=True)
                p = run("cargo", "update", "-p", f"{n}@{v}", "--precise", m)
                print(p.stderr.strip(), flush=True)
                if p.returncode != 0:
                    rejected.append((n, v, m))
            if len(rejected) == len(failed):
                break
            failed = rejected
        fallback_notes += [(n, v, f"`--precise {m}` did not resolve") for n, v, m in failed]
        for n, v, why in fallback_notes:
            print(f"+ cargo update -p {n}@{v}   (fallback: {why})", flush=True)
            p = run("cargo", "update", "-p", f"{n}@{v}")
            print(p.stderr.strip(), flush=True)
        after = audit()
        changed = open("Cargo.lock").read() != original
        old = {(k[1], k[2]) for k in map(key, before)}
        # An advisory on a package bumped to a release that is still vulnerable is progress, not a fix.
        partial = {(k[0], k[1]) for k in map(key, after) if (k[1], k[2]) not in old}
        fixed = {k for k in map(key, before) if k not in set(map(key, after)) and k[:2] not in partial}
        out = []
        if changed:
            out.append("Bumps vulnerable packages in `Cargo.lock` to patched releases "
                       "(`cargo update -p NAME@VERSION --precise MIN_PATCHED`: the lowest patched release, "
                       "semver-compatible only).\n")
            out += [f"- fixed {i}: {n} {v}" for i, n, v in sorted(fixed)]
            out += [f"- fell back to the newest compatible release for {n} {v} ({why}); review that bump by hand"
                    for n, v, why in fallback_notes]
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
