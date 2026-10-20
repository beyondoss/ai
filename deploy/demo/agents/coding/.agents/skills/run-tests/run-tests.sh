#!/usr/bin/env bash
# Run the app's unittest suite and end with a one-line verdict (see SKILL.md).
cd "$(dirname "$0")/../../../app" || exit 2
out=$(python3 -m unittest -v 2>&1)
status=$?
printf '%s\n' "$out"
ran=$(printf '%s\n' "$out" | sed -n 's/^Ran \([0-9]*\) test.*/\1/p')
failed=$(printf '%s\n' "$out" | grep -c '\.\.\. FAIL$')
errors=$(printf '%s\n' "$out" | grep -c '\.\.\. ERROR$')
echo "RUN-TESTS-SKILL: ${ran:-0} run, $failed failed, $errors errors"
exit $status
