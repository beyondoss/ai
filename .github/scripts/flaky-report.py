#!/usr/bin/env python3
"""Make a pass-on-retry visible: report every FLAKY test in nextest JUnit reports.

The `ci` nextest profile retries a failed test once (.config/nextest.toml). A test that fails and
then passes leaves the run green, so without this a flaky test is only a `TRY 2 PASS` line deep in
a log nobody opens. nextest records each failed attempt of a test that finally passed as a
`<flakyFailure>` (or `<flakyError>`) child of its `<testcase>`; this finds them and, for each one,
emits a GitHub warning annotation (shown on the PR's checks page and the run summary) and writes a
count and list to the job summary. It never fails the job: retrying stays the policy, hiding it
does not.

usage: flaky-report.py LABEL JUNIT.xml [JUNIT.xml ...]

A missing report is noted, not an error: the run it would describe already failed or never ran.
"""

import os
import sys
import xml.etree.ElementTree as ET

FLAKY_TAGS = ("flakyFailure", "flakyError")


def escape(text: str) -> str:
    # Workflow-command data escaping (GitHub's `escapeData`).
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def flaky_tests(path: str):
    for case in ET.parse(path).getroot().iter("testcase"):
        failed = [child for tag in FLAKY_TAGS for child in case.findall(tag)]
        if failed:
            first = (failed[0].get("message") or "").strip()
            yield f"{case.get('classname', '')} {case.get('name', '')}".strip(), len(failed), first


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    label, paths = sys.argv[1], sys.argv[2:]
    found, missing = [], []
    for path in paths:
        if not os.path.exists(path):
            missing.append(path)
            continue
        found.extend(flaky_tests(path))

    for name, failures, message in found:
        detail = f"{name} passed only on retry ({failures} failed attempt(s))"
        if message:
            detail += f": {message}"
        print(f"::warning title=Flaky test ({escape(label)})::{escape(detail)}")
    for path in missing:
        print(f"flaky-report: no JUnit report at {path} (the run did not finish)")

    lines = [f"### Flaky tests — {label}: {len(found)}", ""]
    if found:
        lines.append("Passed only after nextest's retry. Green here does not mean stable.")
        lines.append("")
        lines.extend(f"- `{name}` — {failures} failed attempt(s)" for name, failures, _ in found)
    else:
        lines.append("None: every test that ran passed on its first attempt.")
    if missing:
        lines.append("")
        lines.extend(f"- no JUnit report at `{path}` (the run did not finish)" for path in missing)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    text = "\n".join(lines) + "\n"
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    print(f"flaky-report: {len(found)} flaky test(s) in {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
