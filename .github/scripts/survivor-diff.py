#!/usr/bin/env python3
# Turn a cargo-mutants survivor list (file:line:col: desc) into a unified diff that marks exactly
# those lines as added, for `cargo mutants --in-diff`. Usage: survivor-diff.py SRC survivors.txt > d.diff
import sys, collections, difflib
src, surv = sys.argv[1], sys.argv[2]
lines = collections.defaultdict(set)
for l in open(surv):
    l = l.strip()
    if not l: continue
    f, ln = l.split(':')[0], int(l.split(':')[1])
    lines[f].add(ln)
for f, lns in sorted(lines.items()):
    new = open(f"{src}/{f}").read().splitlines(keepends=True)
    old = [x for i, x in enumerate(new, 1) if i not in lns]
    sys.stdout.writelines(difflib.unified_diff(old, new, f"a/{f}", f"b/{f}", n=0))
