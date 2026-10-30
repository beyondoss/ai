#!/usr/bin/env bash
# Print Eve tools parity scorecard from vitest (pass = claimed; fail = gap).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
echo "Eve tools parity — source: https://eve.dev/docs/tools"
echo "Suite: __tests__/eve-parity.test.ts"
echo
npx vitest run __tests__/eve-parity.test.ts --reporter=verbose 2>&1 \
  | rg -e '✓|×|↓|FAIL|PASS|expected fail|Test Files|Tests ' \
  || true
echo
echo "Legend: ✓ proven against Eve docs · ↓/expected fail = not parity yet (honest gap)"
