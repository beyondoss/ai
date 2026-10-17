#!/usr/bin/env python3
"""Fail when a crate's src/ file is missing from its ARCHITECTURE.md module table.

A module table that silently stops listing new files reads as complete and is not. For every
crates/<name>/ARCHITECTURE.md, this finds the module table (the `## Package Structure` or
`## Modules` section) and checks that every `src/**/*.rs` file of that crate is named in the
first cell of one of its rows. A first cell names a file by any backticked token that is:

  - its path, with or without the leading `src/`   (`src/serve_ws.rs`, `dialect/openai.rs`)
  - its module path, without `.rs`                 (`proxy`, `usage/vendor`, `usage::vendor`)
  - a directory that contains it, ending in `/`    (`src/oauth/` covers `src/oauth/pkce.rs`)
  - a glob that matches it                         (`src/tools/mcp_*.rs`)

`x/mod.rs` is also named by `x`. Test-only siblings (`*_tests.rs`, `tests.rs`) are exempt: they
are the tests of the module they sit next to, not modules of their own.

usage: check-module-tables.py [REPO_ROOT]   (default: the repository this script lives in)
"""

import fnmatch
import pathlib
import re
import sys

SECTION = re.compile(r"^##\s+(Package Structure|Modules)\b", re.IGNORECASE)
TOKEN = re.compile(r"`([^`]+)`")


def module_table_cells(doc: str):
    """First cells of every table row in the module-table section(s)."""
    inside = False
    for line in doc.splitlines():
        if line.startswith("## ") or line.startswith("# "):
            inside = bool(SECTION.match(line))
            continue
        if inside and line.startswith("|"):
            cells = line.split("|")
            if len(cells) > 2:
                yield cells[1]


def names(cell: str):
    for token in TOKEN.findall(cell):
        for part in token.split(","):
            part = part.strip()
            if part:
                yield part.removeprefix("src/").replace("::", "/")


def covered(rel: str, tokens) -> bool:
    stem = rel.removesuffix(".rs")
    forms = {rel, stem}
    if rel.endswith("/mod.rs"):
        forms.add(rel.removesuffix("/mod.rs"))
    for t in tokens:
        if t in forms:
            return True
        if t.endswith("/") and rel.startswith(t):
            return True
        if any(c in t for c in "*?[") and fnmatch.fnmatchcase(rel, t):
            return True
    return False


def exempt(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name.endswith("_tests.rs") or name == "tests.rs"


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parents[2])
    failures = 0
    for doc_path in sorted(root.glob("crates/*/ARCHITECTURE.md")):
        crate = doc_path.parent
        tokens = [t for cell in module_table_cells(doc_path.read_text()) for t in names(cell)]
        rel_doc = doc_path.relative_to(root)
        if not tokens:
            print(f"{rel_doc}: no `## Package Structure` or `## Modules` table found")
            failures += 1
            continue
        src = crate / "src"
        for path in sorted(src.rglob("*.rs")):
            rel = path.relative_to(src).as_posix()
            if not exempt(rel) and not covered(rel, tokens):
                print(f"{rel_doc}: module table does not list src/{rel}")
                failures += 1
    if failures:
        print(f"\n{failures} missing. Add a row (first cell: the file, in backticks) to the table.")
        return 1
    print("module tables: every src file is listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
