#!/usr/bin/env python3
"""Red runs for the `list`/`read` answer differential (`jobs::answer_tests`).

Each mutant changes one property of the wire the two ops answer with — a field dropped, a key
renamed, two keys reordered, a value retyped, a number reformatted, a `null` lost, the `more` count
dropped, `stderr_ranges` sent when it is empty — and the differential must go red under every one of
them. A mutant that stays green means the fixture matrix or the comparison does not cover that
property, and the answer's bytes are no longer pinned.

    python3 crates/execd/scripts/answer-mutants.py [--only NAME]

The source file is edited in place and restored from an in-memory copy afterwards (never through
`git checkout`, which has reverted a mutation before it ran); a restore is checked byte for byte, and
the working tree must be clean of the script's own edits when it exits.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parent.parent
ANSWERS = CRATE / "src" / "answer.rs"
TESTS = CRATE / "src" / "jobs.rs"
CARGO = ["cargo", "test", "-p", "beyond-execd", "--lib", "answer_tests"]

PTY = (
    """#[derive(Serialize)]
pub(crate) struct Pty {
    pub cols: u16,
    pub open: bool,
    pub rows: u16,
}""",
    """#[derive(Serialize)]
pub(crate) struct Pty {
    pub cols: u16,
    #[serde(serialize_with = "as_number")]
    pub open: bool,
    pub rows: u16,
}

fn as_number<S: Serializer>(v: &bool, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u8(u8::from(*v))
}""",
)

# (name, file, what it breaks, old bytes, new bytes)
MUTANTS = [
    (
        "drops-a-field",
        ANSWERS,
        "`label` is left out of every row",
        "    pub label: &'a str,\n",
        "    #[serde(skip)]\n    pub label: &'a str,\n",
    ),
    (
        "renames-a-key",
        ANSWERS,
        "`outputStart` is written as `output_start`",
        '    #[serde(rename = "outputStart")]\n',
        "",
    ),
    (
        "reorders-keys",
        ANSWERS,
        "`ended` is written before `command`",
        "    pub command: &'a str,\n    /// When the job ended, 0 until it has.\n    pub ended: u64,\n",
        "    /// When the job ended, 0 until it has.\n    pub ended: u64,\n    pub command: &'a str,\n",
    ),
    (
        "retypes-a-value",
        ANSWERS,
        "`pty.open` is written as 1/0, not true/false",
        *PTY,
    ),
    (
        "reformats-a-number",
        ANSWERS,
        "`more` is written as a float",
        "        serde_json::to_writer(&mut buf, &more)?;",
        "        serde_json::to_writer(&mut buf, &(more as f64))?;",
    ),
    (
        "drops-the-count",
        ANSWERS,
        "`more` is never written",
        "    if more > 0 {\n        buf.extend_from_slice(b\",\\\"more\\\":\");",
        "    if false {\n        buf.extend_from_slice(b\",\\\"more\\\":\");",
    ),
    (
        "loses-a-null",
        ANSWERS,
        "a running job's `code` is absent instead of null",
        "    /// The shell's exit code, `null` until it has one.\n    pub code: Option<i32>,\n",
        "    #[serde(skip_serializing_if = \"Option::is_none\")]\n    pub code: Option<i32>,\n",
    ),
    (
        "empty-ranges-sent",
        ANSWERS,
        "`stderr_ranges` is sent even when the window holds no stderr byte",
        '    #[serde(skip_serializing_if = "<[(u64, u64)]>::is_empty")]\n',
        "",
    ),
]


def run(cmd):
    return subprocess.run(cmd, cwd=CRATE.parent.parent, capture_output=True, text=True)


def failures(out):
    """The tests the run reported as failing, and whether it failed to build instead."""
    if "error[E" in out or "error: could not compile" in out:
        return None, True
    names = [line.split()[1] for line in out.splitlines() if line.startswith("test ") and "FAILED" in line]
    return names, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    base = run(CARGO)
    if base.returncode != 0:
        print("the differential is not green before any mutant, so a red run proves nothing:\n" + base.stdout[-3000:])
        return 1
    print("baseline: green")

    red, bad = 0, []
    for name, path, what, old, new in MUTANTS:
        if a.only and a.only != name:
            continue
        src = path.read_bytes()
        digest = hashlib.sha256(src).hexdigest()
        text = src.decode()
        if text.count(old) != 1:
            bad.append((name, f"ANCHOR-MISSING ({text.count(old)} matches)"))
            path.write_bytes(src)
            continue
        try:
            path.write_text(text.replace(old, new, 1))
            got = run(CARGO)
        finally:
            path.write_bytes(src)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                print(f"!! {path} was not restored; restore it from git before anything else")
                return 2
        names, broken = failures(got.stdout + got.stderr)
        if broken:
            bad.append((name, "BUILD-ERROR (the mutation does not compile)"))
        elif got.returncode == 0:
            bad.append((name, "STAYED GREEN (the differential does not cover this property)"))
        else:
            red += 1
            print(f"red under {name}: {what} -> {', '.join(names) or 'a failing test'}")

    total = len(MUTANTS) if not a.only else 1
    print(f"\n{red}/{total} mutants red")
    for name, why in bad:
        print(f"  {name}: {why}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
