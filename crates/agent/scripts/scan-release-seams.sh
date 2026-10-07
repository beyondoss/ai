#!/bin/sh
# Fail if any given build artifact carries a test seam's name.
#
# A test seam (`BEYOND_AI_AGENT_TEST_SLOW_CHECKPOINT_MS`, ...) changes how the binary behaves when an
# environment variable is set: fine in a debug test binary, never in a shipped one. Every seam is
# compiled only under `#[cfg(debug_assertions)]`, so a release binary must not contain the prefix at
# all. This scans the artifact itself: whatever the source looks like (a name assembled with
# `concat!`, one the source checker in `tests/release_seams.rs` misreads), a name the compiler kept
# is a contiguous string in the binary, and this finds it. CI runs it on every release binary the
# `build (release)` job produces (`mise check:release-seams`).
#
# Usage: scan-release-seams.sh <artifact>...
set -eu
if [ "$#" -eq 0 ]; then
    echo "usage: $0 <artifact>..." >&2
    exit 2
fi
status=0
for artifact in "$@"; do
    if grep -aq 'BEYOND_AI_AGENT_TEST_' "$artifact"; then
        echo "$artifact carries test seam names (they must exist only in debug builds):" >&2
        grep -ao 'BEYOND_AI_AGENT_TEST_[A-Z0-9_]*' "$artifact" | sort -u | sed 's/^/  /' >&2
        status=1
    fi
done
exit "$status"
